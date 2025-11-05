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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import seaborn as sns
        
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

        self.ignore_label = None

        self.preprocessed_dataset_folder_base = join(self.plans['data_preprocessed'], self.plans['dataset_name'])

        self.output_folder_base = join(self.plans['exp_results'], self.plans['dataset_name'],
                                       self.__class__.__name__ + '__' + self.plans['plans_name']) 
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.plans['data_identifier'])

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

        self.num_input_channels = None
        self.network = None
        self.optimizer = self.lr_scheduler = None
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None
        self.loss_recon = None
        self.loss_seg = None
        

        self.normal_weight = 0

        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = Logger()

        current_script = os.path.abspath(sys.argv[0])
        destination = os.path.join(self.output_folder_base, os.path.basename(current_script))
        shutil.copy(current_script, destination)

        self.dataloader_train = self.dataloader_val = self.dataloader_real = self.dataloader_vessel = None

        self._best_ema = None

        self.inference_allowed_mirroring_axes = (0,1,2)

        self.save_every = 2
        self.disable_checkpointing = False

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Trainer has been built."
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


            param_dis = {'dim': 128,
                    'norm': 'none',
                    'activ': 'relu',
                    'n_layer': 4,
                    'gan_type': 'lsgan',
                    'num_scales': 3,
                    'pad_type': 'reflect',

                    }
            self.dis = Discriminator(input_dim=1, params=param_dis).to(self.device)
            
            self.optimizer_dis = torch.optim.Adam(self.dis.parameters(), lr=self.initial_lr, betas=(self.plans['beta1'], self.plans['beta2']),
                                       weight_decay=self.weight_decay)
            
            self.lr_scheduler_dis = PolyLRScheduler(self.optimizer_dis, self.initial_lr, self.num_epochs)
            
            
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
            plt.plot(epochs, loss_real_back, label='Real Background Loss')
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

        if self.is_ddp:
            self.network.module.load_state_dict(new_state_dict)
        else:
            self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
    
    def perform_validation(self, datapath):
        transform = transforms.Compose([
            transforms.Resize((self.plans['imgSize'], self.plans['imgSize'])),
            transforms.ToTensor()
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

                data1 = image.to(self.device, non_blocking=True)
                rec,_ = self.network(data1,data1)

                for i in range(rec.shape[0]):
                    ori_image = data1[i].squeeze().cpu().numpy()
                    if not self.plans['neighbor_num'] == 1:
                        ori_image = ori_image[self.plans['neighbor_num']//2]
                    
                    rec_image = rec[i].squeeze().cpu().numpy()
                    

                    difference =  (rec_image - ori_image)*255

                    ori_image = (ori_image * 255).astype(np.uint8)
                    rec_image = (rec_image * 255).astype(np.uint8)

                    vessel = ((255 - np.clip(difference/0.5, 0, 255))).astype(np.uint8)

                    self.save_output(datapath,ori_image, rec_image, vessel, vessel, image_path[0])

    def perform_validation_matrics(self, datapath, GTPtah, keyslice,imageSize):
        transform = transforms.Compose([
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor()
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

                data1 = image.to(self.device, non_blocking=True)
                rec,_ = self.network(data1,data1)

                for i in range(rec.shape[0]):
                    ori_image = data1[i].squeeze().cpu().numpy()
                    if not self.plans['neighbor_num'] == 1:
                        ori_image = ori_image[self.plans['neighbor_num']//2]
                    ori_image = (ori_image * 255).astype(np.uint8)

                    rec_image = rec[i].squeeze().cpu().numpy()
                    rec_image = (rec_image * 255).astype(np.uint8)

                    difference =  rec_image.astype(np.int16) - ori_image.astype(np.int16)

                    min_value = np.min(difference)
                    max_value = np.max(difference)

                    normalized_diff = (difference - min_value) / (max_value - min_value + 1e-8)

                    normalized_diff = np.clip(normalized_diff, 0, 255)

                    vessel = ((255 - np.clip(difference, 0, 255))).astype(np.uint8)

                    vessel_residual = ((255 - np.clip(difference/0.3, 0, 255))).astype(np.uint8)

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

    
    def postprocess_mask(self, pred_bin, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)

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

            frangi_img = skimage.filters.sato(pred_bin, sigmas=np.linspace(2, 10, 10), black_ridges=False)


            pred_bin = (frangi_img > 0.1) & (pred_avg_uint8 > 20)

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

            gt_flat = gt.flatten()
            pred_flat = pred_bin.flatten()

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
                'recall': recall,
                'specificity': specificity,
                'accuracy': accuracy
            })

        avg = {
            'dice': np.mean([m['dice'] for m in metrics]),
            'recall': np.mean([m['recall'] for m in metrics]),
            'specificity': np.mean([m['specificity'] for m in metrics]),
            'accuracy': np.mean([m['accuracy'] for m in metrics])
        }
        std = {
            'dice': np.std([m['dice'] for m in metrics]),
            'recall': np.std([m['recall'] for m in metrics]),
            'specificity': np.std([m['specificity'] for m in metrics]),
            'accuracy': np.std([m['accuracy'] for m in metrics])
        }
        print("\n Average Metrics:")
        for k in avg:
            self.print_to_log_file(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

        return avg['dice']


    def save_output(self, dataPath, ori, rec, output_norm, output_ori, image_path):
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
        
    def visualization(self, savedir):
        datapath_synthetic = 'datasets/InternalVesselOrganized/Images'
        datapath_real = 'datasets/External_30XCA/Images'
        
        transform = transforms.Compose([
            transforms.Resize((self.plans['imgSize'], self.plans['imgSize'])),
            transforms.ToTensor()
        ])
        
        dataset_s = ImageDataset(
            image_dir=datapath_synthetic,
            transform=transform,
            n=self.plans['neighbor_num']//2,
        )
        dataloader_s = DataLoader(dataset_s, batch_size=1, shuffle=False)
        
        dataset_r = ImageDataset(
            image_dir=datapath_real,
            transform=transform,
            n=self.plans['neighbor_num']//2,
        )
        dataloader_r = DataLoader(dataset_r, batch_size=1, shuffle=False)
        
        len_synthetic = len(dataset_s)
        len_real = len(dataset_r)
        
        print(f"Dataset sizes - Synthetic: {len_synthetic}, Real: {len_real}")
        
        MAX_SAMPLES_PER_DATASET = 500
        
        if len_synthetic <= MAX_SAMPLES_PER_DATASET and len_real <= MAX_SAMPLES_PER_DATASET:
            min_samples = min(len_synthetic, len_real)
            n_synthetic = min_samples
            n_real = min_samples
            print(f"Balanced sampling: using {min_samples} samples from each dataset")
        else:
            n_synthetic = min(len_synthetic, MAX_SAMPLES_PER_DATASET)
            n_real = min(len_real, MAX_SAMPLES_PER_DATASET)
            print(f"Large dataset sampling: Synthetic={n_synthetic}, Real={n_real}")
        
        all_features = []
        all_labels = []
        image_paths = []
        
        print("Extracting features from synthetic data...")
        with torch.no_grad():
            self.network.eval()
            
            for i, (image, image_path) in enumerate(dataloader_s):
                if i >= n_synthetic:
                    break
                    
                print(f'Processing synthetic {i+1}/{n_synthetic}: {os.path.basename(image_path[0])}')
                
                data1 = image.to(self.device, non_blocking=True)
                rec, features = self.network.predict(data1, data1)
                
                feat = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                
                all_features.append(feat.cpu().numpy())
                all_labels.append(0)
                image_paths.append(image_path[0])
            
            print("Extracting features from real data...")
            for i, (image, image_path) in enumerate(dataloader_r):
                if i >= n_real:
                    break
                    
                print(f'Processing real {i+1}/{n_real}: {os.path.basename(image_path[0])}')
                
                data1 = image.to(self.device, non_blocking=True)
                rec, features = self.network.predict(data1, data1)
                
                feat = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                
                all_features.append(feat.cpu().numpy())
                all_labels.append(1)
                image_paths.append(image_path[0])
        
        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        
        print(f"Total samples: {len(all_features)} (Synthetic: {sum(all_labels==0)}, Real: {sum(all_labels==1)})")
        print(f"Feature dimension: {all_features.shape[1]}")
        
        synthetic_features = all_features[all_labels == 0]
        real_features = all_features[all_labels == 1]
        
        mmd_scores = compute_multiple_mmd_scores(synthetic_features, real_features)
        
        main_mmd = mmd_scores.get('RBF (σ=auto)', 0.0)
        
        print("Running t-SNE...")
        n_samples = len(all_features)
        
        perplexity = min(50, max(5, n_samples//8))
        
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity,
            n_iter=2000,
            early_exaggeration=12.0,
            learning_rate=200.0,
            min_grad_norm=1e-7,
            verbose=1
        )
        features_2d = tsne.fit_transform(all_features)
        
        print(f"Using perplexity: {perplexity}")
        
        plt.figure(figsize=(12, 10))
        
        colors = ['#FF6B6B', '#4ECDC4']
        labels = ['Synthetic', 'Real']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = all_labels == i
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1], 
                c=color, 
                alpha=0.9,
                label=f'{label} (n={sum(mask)})', 
                s=100,
                edgecolors='none',
                rasterized=True
            )
        
        plt.xlabel('t-SNE Component 1', fontsize=18)
        plt.ylabel('t-SNE Component 2', fontsize=18)
        plt.title('t-SNE Visualization of Synthetic vs Real Data Features', fontsize=18, fontweight='bold')
        plt.legend(fontsize=18, framealpha=0.9)
        
        plt.grid(False)
        
        x_margin = (features_2d[:, 0].max() - features_2d[:, 0].min()) * 0.05
        y_margin = (features_2d[:, 1].max() - features_2d[:, 1].min()) * 0.05
        plt.xlim(features_2d[:, 0].min() - x_margin, features_2d[:, 0].max() + x_margin)
        plt.ylim(features_2d[:, 1].min() - y_margin, features_2d[:, 1].max() + y_margin)
        
        synthetic_center = features_2d[all_labels == 0].mean(axis=0)
        real_center = features_2d[all_labels == 1].mean(axis=0)
        center_distance = np.linalg.norm(synthetic_center - real_center)
        
        info_text = f'MMD: {main_mmd:.4f}'
        plt.text(0.7, 0.1, info_text, 
                transform=plt.gca().transAxes, fontsize=18,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        
        save_dir = savedir 
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, 'tsne_synthetic_vs_real.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to: {save_path}")
        
        save_path_high = os.path.join(save_dir, 'tsne_synthetic_vs_real_high_res.pdf')
        plt.savefig(save_path_high, dpi=300, bbox_inches='tight', format='pdf')
        print(f"High resolution plot saved to: {save_path_high}")
        
        stats = {
            'total_samples': len(all_features),
            'synthetic_samples': sum(all_labels == 0),
            'real_samples': sum(all_labels == 1),
            'feature_dimension': all_features.shape[1],
            'center_distance': center_distance,
            'synthetic_center': synthetic_center.tolist(),
            'real_center': real_center.tolist(),
            'mmd_scores': mmd_scores,
            'main_mmd_rbf_auto': main_mmd
        }
        
        import json
    
        safe_stats = safe_json_convert(stats)
        
        stats_path = os.path.join(save_dir, 'tsne_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(safe_stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")
        print(f"Main MMD Score (RBF auto): {main_mmd:.6f}")
        
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = all_labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=color, alpha=0.8, label=label, s=60, edgecolors='none')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Standard View')
        plt.legend()
        x_margin = (features_2d[:, 0].max() - features_2d[:, 0].min()) * 0.05
        y_margin = (features_2d[:, 1].max() - features_2d[:, 1].min()) * 0.05
        plt.xlim(features_2d[:, 0].min() - x_margin, features_2d[:, 0].max() + x_margin)
        plt.ylim(features_2d[:, 1].min() - y_margin, features_2d[:, 1].max() + y_margin)
        
        plt.subplot(1, 3, 2)
        from scipy.stats import gaussian_kde
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = all_labels == i
            data = features_2d[mask]
            if len(data) > 10:
                kde = gaussian_kde(data.T)
                
                x_min, x_max = data[:, 0].min(), data[:, 0].max()
                y_min, y_max = data[:, 1].min(), data[:, 1].max()
                xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = np.reshape(kde(positions), xx.shape)
                
                plt.contour(xx, yy, density, colors=color, alpha=0.7, linewidths=2)
                plt.contourf(xx, yy, density, alpha=0.3, colors=[color])
            
            plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.6, s=20, label=label)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Density View')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        
        synthetic_data = features_2d[all_labels == 0]
        real_data = features_2d[all_labels == 1]
        
        plt.hexbin(features_2d[:, 0], features_2d[:, 1], 
                C=all_labels, gridsize=25, cmap='RdYlBu', alpha=0.8, mincnt=1)
        plt.colorbar(label='Data Type (0=Synthetic, 1=Real)')
        
        plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], 
                c='red', alpha=0.3, s=10, label='Synthetic')
        plt.scatter(real_data[:, 0], real_data[:, 1], 
                c='blue', alpha=0.3, s=10, label='Real')
        
        info_text = f'MMD: {main_mmd:.4f}'
        plt.text(0.7, 0.06, info_text, 
                transform=plt.gca().transAxes, fontsize=18,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Hexbin Density View')
        plt.legend()
        
        plt.tight_layout()
        
        dense_path = os.path.join(save_dir, 'tsne_dense_visualization.png')
        plt.savefig(dense_path, dpi=300, bbox_inches='tight')
        print(f"Dense visualization saved to: {dense_path}")
        
        plt.figure(figsize=(10, 8))
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = all_labels == i
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1], 
                c=color, 
                alpha=0.9,
                label=f'{label} (n={sum(mask)})', 
                s=100,
                edgecolors='none'
            )
        
        plt.xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
        plt.ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')
        plt.title('Dense t-SNE Visualization', fontsize=18, fontweight='bold')
        plt.legend(fontsize=18, framealpha=0.9)
        
        x_margin = (features_2d[:, 0].max() - features_2d[:, 0].min()) * 0.02
        y_margin = (features_2d[:, 1].max() - features_2d[:, 1].min()) * 0.02
        plt.xlim(features_2d[:, 0].min() - x_margin, features_2d[:, 0].max() + x_margin)
        plt.ylim(features_2d[:, 1].min() - y_margin, features_2d[:, 1].max() + y_margin)
        
        plt.gca().set_facecolor('#f8f8f8')

        info_text = f'MMD: {main_mmd:.4f}'
        plt.text(0.03, 0.08, info_text, 
                transform=plt.gca().transAxes, fontsize=18,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        
        super_dense_path = os.path.join(save_dir, 'tsne_super_dense.png')
        plt.savefig(super_dense_path, dpi=300, bbox_inches='tight')
        print(f"Super dense visualization saved to: {super_dense_path}")
        
        return features_2d, all_labels, stats

def safe_json_convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(v) for v in obj]
    else:
        return obj
    
def compute_multiple_mmd_scores(synthetic_features, real_features):
    scores = {}
    
    print("Computing MMD scores...")
    
    kernels = [
        ('RBF (σ=1.0)', 'rbf', 1.0),
        ('RBF (σ=auto)', 'rbf', 'auto'),
        ('RBF (σ=0.5)', 'rbf', 0.5),
        ('RBF (σ=2.0)', 'rbf', 2.0),
        ('Linear', 'linear', None),
        ('Polynomial', 'poly', None)
    ]
    
    for name, kernel_type, sigma in kernels:
        try:
            if sigma is not None and kernel_type == 'rbf':
                mmd = compute_mmd(synthetic_features, real_features, kernel=kernel_type, sigma=sigma)
            else:
                mmd = compute_mmd(synthetic_features, real_features, kernel=kernel_type)
            scores[name] = mmd
            print(f"  {name}: {mmd:.6f}")
        except Exception as e:
            print(f"  {name}: Failed ({e})")
            scores[name] = None
    
    return scores

def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    y = torch.tensor(y, dtype=torch.float32) if not isinstance(y, torch.Tensor) else y
    
    def gaussian_kernel(x, y, sigma):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input / (2 * sigma ** 2))
    
    def linear_kernel(x, y):
        return torch.mm(x, y.t())
    
    def polynomial_kernel(x, y, degree=2):
        return (torch.mm(x, y.t()) + 1).pow(degree)
    
    if kernel == 'rbf':
        if sigma == 'auto':
            all_distances = []
            sample_size = min(100, x.size(0), y.size(0))
            x_sample = x[:sample_size]
            y_sample = y[:sample_size]
            
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    all_distances.append(torch.norm(x_sample[i] - x_sample[j]).item())
                    all_distances.append(torch.norm(y_sample[i] - y_sample[j]).item())
            
            sigma = np.median(all_distances)
            print(f"Auto-selected sigma: {sigma:.4f}")
        
        kernel_func = lambda a, b: gaussian_kernel(a, b, sigma)
    elif kernel == 'linear':
        kernel_func = linear_kernel
    elif kernel == 'poly':
        kernel_func = polynomial_kernel
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    xx = kernel_func(x, x).mean()
    yy = kernel_func(y, y).mean()
    xy = kernel_func(x, y).mean()
    
    mmd_squared = xx + yy - 2 * xy
    return torch.sqrt(torch.clamp(mmd_squared, min=0.0)).item()


if __name__ =='__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    plans = load_json('config.json')
    fold = 0
    myTrainer = trainingPlanner(plans, fold)

    path = 'weights/checkpoint_DOSTANet.pth'
    myTrainer.load_checkpoint(path)

    myTrainer.visualization('visualization_results')