
from typing import Tuple, Union, List
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import numpy as np
import os, sys
from utils import get_allowed_n_proc_DA, Logger, dummy_context,empty_cache,collate_outputs
import torch
from torch.cuda.amp import GradScaler
from datetime import datetime
from time import time, sleep
from loss.losse import PolyLRScheduler
from torch import autocast, nn
from torch import distributed as dist
from loss.dice import get_tp_fp_fn_tn
# from torch._dynamo import OptimizedModule
import random
from models.model import Unet,Discriminator,DOSTANet
import cv2
from dataset import PairedDataset,realVesselDataset,realImageDataset,ImageDataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import itertools
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
# import matlab
from loss.dice import SoftDiceLoss
import torch.nn.functional as F
import shutil
from PIL import Image
from glob import glob
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, confusion_matrix
import os
import cv2
import numpy as np
import skimage.filters
from skimage.filters import frangi
from scipy.ndimage import gaussian_filter
from skimage import exposure
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, disk
import numpy as np
from tqdm import tqdm


'''
This trainer is designed for vessel segmentation,
segment vessel and reconstruct orinal image, corresponding to the V1 network that can be found in Whiteboard
'''
def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)

def save_tensor_as_image(tensor, save_path):
    
    tensor = tensor * 255.0  # 归一化到 [0,1]

    images = [torch.cat([tensor[i, c, :, :] for c in range(tensor.shape[1])], dim=1) for i in range(tensor.shape[0])] 
    final_image = torch.cat(images, dim=0)  
    final_image = ((final_image).numpy()).astype(np.uint8)  
    pil_image = Image.fromarray(final_image)
    pil_image.save(save_path)

class CharbonnierLoss(nn.Module):
        """Charbonnier Loss (L1)"""
        def __init__(self, eps=1e-3):
            super(CharbonnierLoss, self).__init__()
            self.eps = eps

        def forward(self, x, y):
            diff = x - y
            # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
            loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
            return loss
        
class trainingPlanner(object):
    def __init__(self, plans: dict, fold: int,
                 device: torch.device = torch.device('cuda')):
        
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device
        self.fold = fold
        self.plans = plans
        self.alpha = 0.999
        self.beta = 0.03
        self.tau = 300
        self._best_dice_E = self._best_dice_I = None

        self.ignore_label = None  #None

        self.preprocessed_dataset_folder_base = join(self.plans['data_preprocessed'], self.plans['dataset_name'])

        self.output_folder_base = join(self.plans['exp_results'], self.plans['dataset_name'],
                                       self.__class__.__name__ + '__' + self.plans['plans_name']) 
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.plans['data_identifier'])

        ### Some hyperparameters for you to fiddle with
        ### Some hyperparameters for you to fiddle with

        self.batch_size = int(self.plans['batch_size'])

        self.labeled_batch = int(self.plans['batch_size'])

        self.initial_lr = float(self.plans['initial_lr'])  
        self.weight_decay = float(self.plans['weight_decay'])  
        self.oversample_foreground_percent = float(self.plans['oversample_foreground_percent'])  
        self.num_iterations_per_epoch = int(self.plans['num_iterations_per_epoch']) 
        self.num_val_iterations_per_epoch = int(self.plans['num_val_iterations_per_epoch'])  
        self.num_epochs = int(self.plans['num_epochs'])  

        self.current_epoch = 0
        self.enable_deep_supervision = False

        ### Dealing with labels/regions
        # self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize
        self.loss_recon = None
        self.loss_seg = None
        

        self.normal_weight = 0

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = Logger()

        current_script = os.path.abspath(sys.argv[0])
        destination = os.path.join(self.output_folder_base, os.path.basename(current_script))
        shutil.copy(current_script, destination)

        ### placeholders
        self.dataloader_train = self.dataloader_val = self.dataloader_real = self.dataloader_vessel = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = (0,1,2)  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 2
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Trainer has been bulit."
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)
        
    def initialize(self):
        if not self.was_initialized:

            empty_cache(self.device)

            self.num_input_channels = 1
            self.network = DOSTANet(in_channels=self.plans['neighbor_num'],is_mixup=self.plans["is_shuffle"],ratio_swap=self.plans['ratio_swap'], is_att=self.plans["is_DTAM"]).to(self.device)

            
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr, betas=(self.plans['beta1'], self.plans['beta2']),
                                       weight_decay=self.weight_decay)
            
            self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.num_epochs)


            param_dis = {'dim': 128,                     # number of filters in the bottommost layer
                    'norm': 'none',                  # normalization layer [none/bn/in/ln]
                    'activ': 'relu',                # activation function [relu/lrelu/prelu/selu/tanh]
                    'n_layer': 4,                  # number of layers in D
                    'gan_type': 'lsgan',             # GAN loss [lsgan/nsgan]
                    'num_scales': 3,               # number of scales
                    'pad_type': 'reflect',           # padding type [zero/reflect]

                    }
            self.dis = Discriminator(input_dim=1, params=param_dis).to(self.device)  # discriminator
            
            self.optimizer_dis = torch.optim.Adam(self.dis.parameters(), lr=self.initial_lr, betas=(self.plans['beta1'], self.plans['beta2']),
                                       weight_decay=self.weight_decay)
            
            self.lr_scheduler_dis = PolyLRScheduler(self.optimizer_dis, self.initial_lr, self.num_epochs)
            
            
            self.loss_recon = self.WeightedL1Loss(weight_1=0.7, weight_0=0.3)  
              #
            
            # self.loss_recon = nn.L1Loss()
            # self.loss_seg = DC_and_CE_loss({'batch_dice': False,
            #                     'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1, ignore_label=self.ignore_label)
            # self.loss_seg = nn.MSELoss()
            
            self.was_initialized = True
            soft_dice_kwargs = {'batch_dice': True,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': False}
            self.dc = nn.MSELoss()

    def plot_loss_curves(self, epoch):
        logger = self.logger.my_fantastic_logging
        plt.figure(figsize=(10, 6))
        
        epochs = list(range(epoch + 1))
        l_rec = logger['train_l_rec'][:epoch+1] if 'train_l_rec' in logger else []
        loss_gen_adv = logger['train_loss_gen_adv'][:epoch+1] if 'train_loss_gen_adv' in logger else []
        loss_real_back = logger['loss_real_background'][:epoch+1] if 'loss_real_background' in logger else []
        loss_real_vessel = logger['loss_real_vessel'][:epoch+1] if 'loss_real_vessel' in logger else []
        loss_DC = logger['loss_DC'][:epoch+1] if 'loss_DC' in logger else []

        if l_rec:
            plt.plot(epochs, l_rec, label='Reconstruction Loss')
        if loss_gen_adv:
            plt.plot(epochs, loss_gen_adv, label='Adversarial Loss') 
        if loss_real_back:
            plt.plot(epochs, loss_real_back, label='Real Bacoground Loss')
        if loss_real_vessel:
            plt.plot(epochs, loss_real_vessel, label='Real Vessel Loss')
        if loss_DC:
            plt.plot(epochs, loss_DC, label='DC Vessel Loss')

            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curves (Epoch {epoch})')
        plt.legend()
        plt.grid(True)
        
        save_dir = join(self.output_folder, 'loss_plots')
        maybe_mkdir_p(save_dir)
        plt.savefig(join(save_dir, f'loss_epoch_{epoch}.png'))
        plt.close()
    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))
    
    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                # if isinstance(mod, OptimizedModule):
                #     mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.plans,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def get_dataloaders(self):
        
        train_path = os.path.join(self.preprocessed_dataset_folder,'Train')
        dataset_Tr = PairedDataset(
            train_path,
            transform=True,
            channel=self.plans['neighbor_num'],
            size=(self.plans['imgSize'],self.plans['imgSize'])
        )
        loader_tr = DataLoader(dataset_Tr, batch_size=self.batch_size, shuffle=True,drop_last=True)

        test_path = os.path.join(self.preprocessed_dataset_folder,'Test')
        dataset_Ts = PairedDataset(
            test_path,
            transform=False,
            size=(self.plans['imgSize'],self.plans['imgSize'])
        )
        loader_ts = DataLoader(dataset_Ts, batch_size=1)

        realImagePath = os.path.join(self.preprocessed_dataset_folder,'realImages')
        dataset_real = realImageDataset(realImagePath,neibor=self.plans['neighbor_num'],size=(self.plans['imgSize'],self.plans['imgSize']))
        loader_real = DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True,drop_last=True)
        data_iter_real = itertools.cycle(iter(loader_real))

        vesselImagePath = os.path.join(self.preprocessed_dataset_folder,'vesselImage')
        dataset_vessel = realVesselDataset(vesselImagePath,size=(self.plans['imgSize'],self.plans['imgSize']))
        loader_vessel = DataLoader(dataset_vessel, batch_size=self.batch_size, shuffle=True,drop_last=True)
        data_iter_vessel = itertools.cycle(iter(loader_vessel))

        return loader_tr, loader_ts, data_iter_real, data_iter_vessel
    

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def train_step(self, batch: dict) -> dict:
        
        data1 = batch['image1']
        # data2 = batch['image2']
        target = batch['image1_ori']

        mask_vessel = batch['label1']
        ori_vessel = batch['label1_ori']

        real_pair = next(self.dataloader_real)
        real_data = real_pair['image']

        
        # print('real_data:',real_data.shape)
        real_devessel = real_pair['image_deVessel']
        real_devessel_forVessel = real_pair['image_deVessel_vessel']
        real_filledVessel = real_pair['image_filled']
        # real_filledVessel_real = real_pair['image_filled_real']

        vessel_data = next(self.dataloader_vessel)

        data1 = data1.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        mask_vessel = mask_vessel.to(self.device, non_blocking=True)
        ori_vessel = ori_vessel.to(self.device, non_blocking=True)
        
        real_data = real_data.to(self.device, non_blocking=True)
        real_devessel = real_devessel.to(self.device, non_blocking=True)
        real_devessel_forVessel = real_devessel_forVessel.to(self.device, non_blocking=True)
        real_filledVessel = real_filledVessel.to(self.device, non_blocking=True)
        # real_filledVessel_real = real_filledVessel_real.to(self.device, non_blocking=True)

        vessel_data = vessel_data.to(self.device, non_blocking=True)

        clean_back = target.repeat(1, data1.size(1), 1, 1)


        # save_tensor_as_image(data1.cpu(), "Synthetic_input_5.png")

         # train Dis first
        

        # target = target.int()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Encode
        # prediction = self.network(torch.cat([data1,real_data]))

        # pre_syth = prediction[:self.batch_size,:]
        # pre_real = prediction[self.batch_size:,:]

        pre_syth, pre_real = self.network(data1, real_data)
        # _, pre_clean_back = self.network(data1, clean_back)

        l_rec = self.loss_recon(pre_syth, target,mask_vessel)
        # l_rec = self.loss_recon(pre_syth, target)

        # l_rec_clean = self.loss_recon(pre_clean_back, target)

        loss_gen_adv = self.dis.calc_gen_loss(torch.cat([pre_syth,pre_real]))
        loss_real_background = self.masked_l1_loss(pre_real,real_devessel)
        loss_real_vessel = self.filled_l1_loss(pre_real, real_filledVessel, real_devessel_forVessel)
        # loss_real_vessel = self.filled_l1_loss(pre_real, real_filledVessel_real, real_devessel_forVessel)

        #Dice loss
        # residual_vessel = F.relu(pre_syth-data1)
        if self.plans['neighbor_num']>1:
            n = data1.shape[1]  
            middle_channel = n // 2 
            residual_vessel = pre_syth-data1[:, middle_channel, :, :].unsqueeze(1)
        loss_dc = self.dc(residual_vessel, ori_vessel)

        weight_loss_real_vessel = self.linear_decay(current_epoch=self.current_epoch,   # 当前 epoch
                            start_weight=1,   
                            end_weight=0.1, 
                            total_epochs=self.num_epochs,   
                            start_epoch=10      
                        )
        l = l_rec + 0.05*loss_gen_adv + weight_loss_real_vessel*loss_real_vessel + loss_real_background  
        # print('Loss:',l_rec.item(), loss_gen_adv.item(), loss_dc.item())
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()


        self.optimizer_dis.zero_grad()
        # encode
        # prediction = self.network(real_data)
        for _ in range(self.plans['D_step']):

            # prediction = self.network(data1)
            # D loss
            loss_dis = self.dis.calc_dis_loss(torch.cat([pre_syth.detach(),pre_real.detach()]), target)
            loss_dis_total = loss_dis
            loss_dis_total.backward()
            self.optimizer_dis.step()

        return {
        'loss': l.detach().cpu().numpy(),
        'l_rec': l_rec.detach().cpu().numpy(),
        'loss_gen_adv': loss_gen_adv.detach().cpu().numpy(),
        'loss_real_background': loss_real_background.detach().cpu().numpy(),
        'loss_real_vessel': loss_real_vessel.detach().cpu().numpy(),
        'loss_DC': loss_dc.detach().cpu().numpy()
    }
    
    def linear_decay(
    self,
    current_epoch: int,
    start_weight: float = 1.0,
    end_weight: float = 0.0001,
    total_epochs: int = 100,
    start_epoch: int = 0
    ) -> float:

        if current_epoch < start_epoch:
            return start_weight
        remaining_epochs = total_epochs - start_epoch     
        if remaining_epochs <= 0:
            return end_weight

        decay_rate = (start_weight - end_weight) / remaining_epochs
        elapsed_epochs = current_epoch - start_epoch
        current_weight = start_weight - decay_rate * elapsed_epochs

        return max(current_weight, end_weight)
    
    class WeightedL1Loss(nn.Module):
        def __init__(self, weight_1=0.6, weight_0=0.4):
            super().__init__()
            self.weight_1 = weight_1
            self.weight_0 = weight_0

        def forward(self, input, target, mask):            
            weights = mask * self.weight_1 + (1 - mask) * self.weight_0
            
            l1_loss = torch.abs(input - target)
            weighted_loss = l1_loss * weights

            return weighted_loss.mean()


    def masked_l1_loss(self,pred, target):

        mask = (target != 0).float()
        loss = torch.abs(pred - target)
        loss_masked = loss * mask

        eps = 1e-8  
        loss_mean = loss_masked.sum() / (mask.sum() + eps)
        return loss_mean

    def filled_l1_loss(self, pred, target, vessel_region):
        mask = (vessel_region == 0).float()
        # mask = ((vessel_region == 0) & (pred < (135/255))).float()

        elementwise_loss = torch.abs(pred - target)

        masked_loss = elementwise_loss * mask

        eps = 1e-8 
        normalized_loss = masked_loss.sum() / (mask.sum() + eps)
        
        return normalized_loss
    
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            # if isinstance(self.network.module, OptimizedModule):
            #     self.network.module._orig_mod.load_state_dict(new_state_dict)
            # else:
            self.network.module.load_state_dict(new_state_dict)
        else:
            # if isinstance(self.network, OptimizedModule):
            #     self.network._orig_mod.load_state_dict(new_state_dict)
            # else:
            self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
    
    def validation_step(self, batch, num):
        data1 = batch['image1']
        # data2 = batch['data2']
        # target = batch['target']
        target = batch['image1_ori']
       
        data1 = data1.to(self.device, non_blocking=True)
       
        # output = self.network(data1.repeat(1, self.plans['neighbor_num'], 1, 1))
        if self.plans['neighbor_num'] != data1.shape[1]:
            dataR = data1.repeat(1, self.plans['neighbor_num'], 1, 1)
        output,_ = self.network(dataR,dataR)

        # output_seg = output

        # rec = rec.squeeze(1)

        for i in range(output.shape[0]):
            image = data1[i].squeeze().cpu().numpy()  
            image = (image * 255).astype(np.uint8) 
        
            recon_ori = output[i].squeeze().cpu().numpy()  
            recon_ori = (recon_ori * 255).astype(np.uint8) 

            difference =  recon_ori.astype(np.int16) - image.astype(np.int16)
            difference = difference/0.3
            # min_value = np.min(difference)
            # max_value = np.max(difference)

            # normalized_diff = (difference - min_value) / (max_value - min_value + 1e-8)

            # normalized_diff = np.clip(normalized_diff, 0, 255)

            # norm_vessel = ((255 - normalized_diff * 255)*0.6).astype(np.uint8)

            vessel = ((255 - np.clip(difference, 0, 255))*0.6).astype(np.uint8)

            if len(target[i].shape) > 1:
                target = target[i].squeeze().numpy()  
                target = (target * 255).astype(np.uint8)  
                combined_image = np.concatenate([image, target, recon_ori,vessel], axis=1)  
            else:
                combined_image = np.concatenate([image, recon_ori,vessel], axis=1)  

            savePath = os.path.join(self.output_folder, str(self.current_epoch))
            os.makedirs(savePath,exist_ok=True)
            cv2.imwrite(os.path.join(savePath, f"reconstruction_{num}.png"), combined_image) 

        del data1,output

        return {'loss':0}
    
    def perform_validation(self, datapath):
        transform = transforms.Compose([
            # transforms.CenterCrop((self.plans['imgSize'], self.plans['imgSize'])),  # Resize to 224x224
            transforms.Resize((self.plans['imgSize'], self.plans['imgSize'])),  # Resize to target size
            transforms.ToTensor()          # Convert to tensor
        ])

        dataset = ImageDataset(
            image_dir=datapath,
            transform=transform,
            n=self.plans['neighbor_num']//2,
        )

        dataloader_test = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            self.network.eval()
            for i, (image, image_path) in enumerate(dataloader_test):
                # print('Processing', os.path.basename(image_path[0]))

                data1 = image.to(self.device, non_blocking=True)
                rec,_ = self.network(data1,data1)
                # rec = rec.squeeze(1)
                for i in range(rec.shape[0]):
                    ori_image = data1[i].squeeze().cpu().numpy()  
                    if not self.plans['neighbor_num'] == 1:
                        ori_image = ori_image[self.plans['neighbor_num']//2]
                    
                    rec_image = rec[i].squeeze().cpu().numpy()  
                    

                    # print('Min and max:',np.min(difference),np.max(difference))
                    difference =  (rec_image - ori_image)*255

                    ori_image = (ori_image * 255).astype(np.uint8)  
                    rec_image = (rec_image * 255).astype(np.uint8) 

                    # difference = difference/0.3

                    # normalized_diff = (difference - min_value) / (max_value - min_value + 1e-8)

                    # normalized_diff = np.clip(normalized_diff, 0, 255)

                    # norm_vessel = ((255 - normalized_diff * 255)*0.6).astype(np.uint8)

                    # vessel = ((255 - np.clip(difference, 0, 255))*0.6).astype(np.uint8)
                    # norm_vessel = ((255 - normalized_diff * 255)).astype(np.uint8)

                    vessel = ((255 - np.clip(difference/0.5, 0, 255))).astype(np.uint8)
                    # print(ori_image.shape, rec_image.shape, norm_vessel.shape, vessel.shape)

                    self.save_output(datapath,ori_image, rec_image, vessel, vessel, image_path[0])

    def perform_validation_matrics(self, datapath, GTPtah, keyslice,imageSize):
        transform = transforms.Compose([
            # transforms.CenterCrop((self.plans['imgSize'], self.plans['imgSize'])),  # Resize to 224x224
            transforms.Resize((imageSize, imageSize)),  # Resize to target size
            transforms.ToTensor()          # Convert to tensor
        ])

        dataset = ImageDataset(
            image_dir=datapath,
            transform=transform,
            n=self.plans['neighbor_num']//2,
        )

        dataloader_test = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            self.network.eval()
            for i, (image, image_path) in enumerate(dataloader_test):
                # print('Processing', os.path.basename(image_path[0]))

                data1 = image.to(self.device, non_blocking=True)
                rec,_ = self.network(data1,data1)
                # rec = rec.squeeze(1)
                for i in range(rec.shape[0]):
                    ori_image = data1[i].squeeze().cpu().numpy() 
                    if not self.plans['neighbor_num'] == 1:
                        ori_image = ori_image[self.plans['neighbor_num']//2]
                    ori_image = (ori_image * 255).astype(np.uint8) 

                    rec_image = rec[i].squeeze().cpu().numpy() 
                    rec_image = (rec_image * 255).astype(np.uint8) 

                    # print('Min and max:',np.min(difference),np.max(difference))
                    difference =  rec_image.astype(np.int16) - ori_image.astype(np.int16)

                    # difference = difference/0.3

                    min_value = np.min(difference)
                    max_value = np.max(difference)

                    normalized_diff = (difference - min_value) / (max_value - min_value + 1e-8)

                    normalized_diff = np.clip(normalized_diff, 0, 255)

                    vessel = ((255 - np.clip(difference, 0, 255))).astype(np.uint8)

                    vessel_residual = ((255 - np.clip(difference/0.3, 0, 255))).astype(np.uint8)
                    # print(ori_image.shape, rec_image.shape, norm_vessel.shape, vessel.shape)

                    res_dir = os.path.join(self.output_folder,"reconstruction", os.path.relpath(image_path[0], datapath))
                    os.makedirs(os.path.dirname(res_dir), exist_ok=True)
                    cv2.imwrite(res_dir, rec_image)  

                    res_dir = os.path.join(self.output_folder,"segmentation", os.path.relpath(image_path[0], datapath))
                    os.makedirs(os.path.dirname(res_dir), exist_ok=True)
                    cv2.imwrite(res_dir, vessel) 

                    res_dir = os.path.join(self.output_folder,"Residual", os.path.relpath(image_path[0], datapath))
                    os.makedirs(os.path.dirname(res_dir), exist_ok=True)
                    cv2.imwrite(res_dir, vessel_residual) 

        return self.evaluate_segmentation(GTPtah, join(self.output_folder,"Residual"), keyslice, binary_save_root=join(self.output_folder,"binary"))

                    # self.save_output(datapath,ori_image, rec_image, norm_vessel, vessel, image_path[0])
    
    def postprocess_mask(self, pred_bin, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)
        # closed = pred_bin

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)

        if num_labels <= 1:
            return closed

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        largest_component = (labels == largest_label).astype(np.uint8)

        return largest_component

    def evaluate_segmentation(self, gt_root, pred_root, selected_indices, binary_save_root=None):
        volumes = sorted(os.listdir(gt_root))
        metrics = []

        for volume in tqdm(volumes):
            gt_dir = os.path.join(gt_root, volume)
            pred_dir = os.path.join(pred_root, volume)

            gt_images = glob(os.path.join(gt_dir, '*.png')) + glob(os.path.join(gt_dir, '*.jpg'))
            if len(gt_images) != 1:
                print(f"⚠️ Warning: {gt_dir} should contain exactly 1 image (png or jpg).")
                continue
            gt = cv2.imread(gt_images[0], cv2.IMREAD_GRAYSCALE)
            gt = (gt > 10).astype(np.uint8)

            pred_images = sorted(glob(os.path.join(pred_dir, '*.png')) + glob(os.path.join(pred_dir, '*.jpg')))
            selected_preds = []

            for i in selected_indices:
                if i < len(pred_images):
                    img = cv2.imread(pred_images[i], cv2.IMREAD_GRAYSCALE)
                    selected_preds.append(img)
                else:
                    print(f"Index {i} out of range in {pred_dir}")

            if not selected_preds:
                continue

            pred_avg = np.min(selected_preds, axis=0)
            pred_avg_uint8 = np.clip(255-pred_avg, 0, 255).astype(np.uint8)

            fixed_thresh = 20
            _, pred_bin = cv2.threshold(pred_avg_uint8, fixed_thresh, 255, cv2.THRESH_BINARY)
            # pred_bin = postprocess_mask(pred_bin, kernel_size=3)


            frangi_img = skimage.filters.sato(pred_bin, sigmas=np.linspace(2, 10, 10), black_ridges=False)


            pred_bin = (frangi_img > 0.1) & (pred_avg_uint8 > 20)

            # pred_bin = binary_fill_holes(pred_bin)

            pred_bin = remove_small_objects(pred_bin, min_size=100)
            
            pred_bin_dilated = dilation(pred_bin, disk(3))  

            label_img = label(pred_bin_dilated)
            regions = regionprops(label_img)

            if regions:
                max_region = max(regions, key=lambda r: r.area)
                largest_cc_mask = label_img == max_region.label
                pred_bin = np.logical_and(pred_bin, largest_cc_mask)
            else:
                print("No max pred_bin")
                                              
            pred_bin = (pred_bin*255).astype(np.uint8)
            pred_bin = cv2.threshold(pred_bin, fixed_thresh, 1, cv2.THRESH_BINARY)[1]

            if pred_bin.shape != gt.shape:
                pred_bin = cv2.resize(pred_bin, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            if binary_save_root:
                save_dir = os.path.join(binary_save_root, volume)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{volume}.png")
                cv2.imwrite(save_path, pred_bin * 255)  

                save_path = os.path.join(save_dir, f"{volume}_pred.png")
                cv2.imwrite(save_path, pred_avg_uint8)  
                save_path = os.path.join(save_dir, f"{volume}_gt.png")
                cv2.imwrite(save_path, gt*255) 

            # --- 5. Flatten for metrics ---
            gt_flat = gt.flatten()
            pred_flat = pred_bin.flatten()

            # --- 6. Compute metrics ---
            dice = f1_score(gt_flat, pred_flat)
            iou = jaccard_score(gt_flat, pred_flat)
            precision = precision_score(gt_flat, pred_flat)
            recall = recall_score(gt_flat, pred_flat)

            tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            metrics.append({
                'volume': volume,
                'dice': dice,
                #'iou': iou,
                #'precision': precision,
                'recall': recall,
                #'dr': recall,  # same as recall
                'specificity': specificity, 
                'accuracy': accuracy         
            })

        avg = {
            'dice': np.mean([m['dice'] for m in metrics]),
            #'iou': np.mean([m['iou'] for m in metrics]),
            #'precision': np.mean([m['precision'] for m in metrics]),
            'recall': np.mean([m['recall'] for m in metrics]),
            #'dr': np.mean([m['dr'] for m in metrics]),
            'specificity': np.mean([m['specificity'] for m in metrics]), 
            'accuracy': np.mean([m['accuracy'] for m in metrics])        
        }
        std = {
            'dice': np.std([m['dice'] for m in metrics]),
            #'iou': np.std([m['iou'] for m in metrics]),
            #'precision': np.std([m['precision'] for m in metrics]),
            'recall': np.std([m['recall'] for m in metrics]),
            #'dr': np.std([m['dr'] for m in metrics]),
            'specificity': np.std([m['specificity'] for m in metrics]),  
            'accuracy': np.std([m['accuracy'] for m in metrics])         
        }
        print("\n Average Metrics:")
        for k in avg:
            self.print_to_log_file(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

        return avg['dice']


    def save_output(self, dataPath, ori, rec, output_norm, output_ori, image_path):
        # Create segmentation folder structure
        segmentation_dir = os.path.join(self.output_folder, "segmentation_norm", os.path.relpath(image_path, dataPath))
        os.makedirs(os.path.dirname(segmentation_dir), exist_ok=True)
        cv2.imwrite(segmentation_dir, output_norm)  

        res_dir = os.path.join(self.output_folder,"reconstruction", os.path.relpath(image_path, dataPath))
        os.makedirs(os.path.dirname(res_dir), exist_ok=True)
        cv2.imwrite(res_dir, rec)  

        res_dir = os.path.join(self.output_folder,"segmentation_ori", os.path.relpath(image_path, dataPath))
        os.makedirs(os.path.dirname(res_dir), exist_ok=True)
        cv2.imwrite(res_dir, output_ori)  

        res_dir = os.path.join(self.output_folder,"CMB", os.path.relpath(image_path, dataPath))
        os.makedirs(os.path.dirname(res_dir), exist_ok=True)

        combined_image = np.concatenate([ori,  rec, output_ori], axis=1) 
        cv2.imwrite(res_dir, combined_image) 
    
    def start_train(self):
        self.dataloader_train, self.dataloader_val, self.dataloader_real, self.dataloader_vessel = self.get_dataloaders()

        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        print(self.plans)

        save_json(self.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.network.train()
            self.lr_scheduler.step(self.current_epoch)

            self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
            self.print_to_log_file('')
            self.print_to_log_file(f'Epoch {self.current_epoch}')
            self.print_to_log_file(
                f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
            
            # lrs are the same for all workers so we don't need to gather them in case of DDP training
            self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

            train_outputs = []
            for i, batch in enumerate(self.dataloader_train):
                if i > self.num_iterations_per_epoch:
                    break
                train_outputs.append(self.train_step(batch))
            outputs = collate_outputs(train_outputs)
            loss_here = np.mean(outputs['loss'])
            l_rec_avg = np.mean(outputs['l_rec'])
            loss_gen_adv_avg = np.mean(outputs['loss_gen_adv'])
            loss_real_recon_avg = np.mean(outputs['loss_real_background'])
            loss_real_vessel_avg = np.mean(outputs['loss_real_vessel'])
            loss_DC_avg = np.mean(outputs['loss_DC'])

            self.logger.log('train_losses', loss_here, epoch)
            self.logger.log('train_l_rec', l_rec_avg, epoch)
            self.logger.log('train_loss_gen_adv', loss_gen_adv_avg, epoch)
            self.logger.log('loss_real_background', loss_real_recon_avg, epoch)
            self.logger.log('loss_real_vessel', loss_real_vessel_avg, epoch)
            self.logger.log('loss_DC', loss_DC_avg, epoch)

            self.plot_loss_curves(epoch)


            # #evalation
            if self.current_epoch % 5 == 0:
                with torch.no_grad():
                    self.network.eval()

                    dataPath = 'datasets/External_30XCA/Images'
                    GTPtah = 'datasets/External_30XCA/GT'
                    self.output_folder = join('outputs', self.plans['plans_name'], 'External_30XCA')
                    dice_E = self.perform_validation_matrics(dataPath, GTPtah, [2], myTrainer.plans['imgSize'])
                    # myTrainer.evaluate_segmentation(GTPtah, join(myTrainer.output_folder,"segmentation"), [2],binary_save_root)

                    dataPath = 'datasets/InternalVesselOrganized/Images'
                    GTPtah = 'datasets/InternalVesselOrganized/GT'
                    self.output_folder = join('outputs', self.plans['plans_name'], 'InternalVesselOrganized')
                    dice_I = self.perform_validation_matrics(dataPath, GTPtah, [0,1,2,3,4,5,6], self.plans['imgSize'])

                    self.print_to_log_file('Val_dice_I', np.round(dice_E, decimals=4))
                    self.print_to_log_file('Val_dice_E', np.round(dice_I, decimals=4))
                        
                    if self._best_dice_I is None or dice_E > self._best_dice_E or dice_I > self._best_dice_I:
                        self._best_dice_E = dice_E
                        self._best_dice_I = dice_I
                        self.print_to_log_file(f"Yayy! New best Dice_E: {np.round(dice_E, decimals=4)}")
                        self.print_to_log_file(f"Yayy! New best Dice_I: {np.round(dice_I, decimals=4)}")
                        self.save_checkpoint(join(self.output_folder, f'checkpoint_best_DICE_{self.current_epoch}.pth'))
                    self.output_folder = join(self.output_folder_base, f'fold_{fold}')
            #     for i, batch in enumerate(self.dataloader_val):
            #         # batch = batch.to(self.device)
            #         val_outputs.append(self.validation_step(batch,i))
            # #     outputs_val = collate_outputs(val_outputs)
            #     tp = np.sum(outputs_val['tp_hard'], 0)
            #     fp = np.sum(outputs_val['fp_hard'], 0)
            #     fn = np.sum(outputs_val['fn_hard'], 0)
            #     loss_here = np.mean(outputs_val['loss'])

            #     global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
            #     mean_fg_dice = np.nanmean(global_dc_per_class)
            #     self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
            #     self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
            #     self.logger.log('val_losses', loss_here, self.current_epoch)

            self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

            self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
            # self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
            # self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
            #                                     self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
            self.print_to_log_file(
                f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

            # handling periodic checkpointing
            current_epoch = self.current_epoch
            if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
                self.save_checkpoint(join(self.output_folder, f'checkpoint_epoch_{self.current_epoch}.pth'))
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

            # # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
            # if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            #     self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            #     self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            #     self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

            # if self.local_rank == 0:
            #     self.logger.plot_progress_png(self.output_folder)

            self.current_epoch += 1

      

        empty_cache(self.device)
        self.print_to_log_file("Training done.")     

    def create_video(self, img_path, output_video, fps=10):
        os.makedirs(os.path.dirname(output_video),exist_ok=True)
        img_files = sorted([f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg'))])

        first_image = cv2.imread(os.path.join(img_path, img_files[0]))
        height, width, _ = first_image.shape
        video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for img_file in img_files:
            img = cv2.imread(os.path.join(img_path, img_file))

            video_writer.write(img)

            # print(f"Processed and added to video: {img_file}")

        video_writer.release()
        print(f"Video saved to {output_video}")

if __name__ =='__main__':
    # os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
# def __init__(self, plans: dict, fold: int, dataset_json: dict,
#                  device: torch.device = torch.device('cuda')):

    plans = load_json('config.json')
    fold = 0
    myTrainer = trainingPlanner(plans, fold)

    myTrainer.start_train()

    dataPath = 'datasets/External_30XCA/Images'
    GTPtah = 'datasets/External_30XCA/GT'
    myTrainer.output_folder = join('outputs', myTrainer.plans['plans_name'], 'External_30XCA')
    myTrainer.perform_validation_matrics(dataPath, GTPtah, [2], myTrainer.plans['imgSize'])
    # myTrainer.evaluate_segmentation(GTPtah, join(myTrainer.output_folder,"segmentation"), [2],binary_save_root)

    dataPath = 'datasets/InternalVesselOrganized/Images'
    GTPtah = 'datasets/InternalVesselOrganized/GT'
    myTrainer.output_folder = join('outputs', myTrainer.plans['plans_name'], 'InternalVesselOrganized')
    myTrainer.perform_validation_matrics(dataPath, GTPtah, [0,1,2,3,4,5,6], myTrainer.plans['imgSize'])

    

    
