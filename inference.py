
from typing import Tuple, Union, List
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import numpy as np
import os, sys
from util.utils import get_allowed_n_proc_DA, Logger, dummy_context,empty_cache,collate_outputs
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
matplotlib.use('Agg')  # 不需要GUI
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
            
            
            # self.loss_recon = self.WeightedL1Loss(weight_1=0.7, weight_0=0.3)  
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

                    rec_image = rec[i].squeeze().cpu().numpy()  # 转为 numpy 格式
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
            pred_bin_s = cv2.threshold(pred_bin, fixed_thresh, 1, cv2.THRESH_BINARY)[1]

            if pred_bin.shape != gt.shape:
                pred_bin = cv2.resize(pred_bin_s, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            if binary_save_root:
                save_dir = os.path.join(binary_save_root, volume)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{volume}.png")
                cv2.imwrite(save_path, pred_bin_s * 255)  

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

    path = 'weights/checkpoint_DOSTANet.pth'
    myTrainer.load_checkpoint(path)

    

    dataPath = 'datasets/External_30XCA/Images'
    GTPtah = 'datasets/External_30XCA/GT'
    myTrainer.output_folder = join('outputs', myTrainer.plans['plans_name'], 'External_30XCA')
    myTrainer.perform_validation_matrics(dataPath, GTPtah, [2], myTrainer.plans['imgSize'])
