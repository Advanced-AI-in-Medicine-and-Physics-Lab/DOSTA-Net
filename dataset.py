from torchvision.transforms import functional as F
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import os
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torchvision.transforms import RandomErasing 
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np


class realVesselDataset(Dataset):
    def __init__(self, root_dir, transform=None,size=(512,512)):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.size = size

        # Collect image file paths recursively
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))

        # Sort image paths to maintain consistent order
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Ensure RGB format

        transform = transforms.Compose([
            transforms.Resize(self.size),  # Apply random cropping
            transforms.ToTensor(),                    # Convert image to tensor
        ])

        image = transform(image)

        return image



class ImageDataset(Dataset):
    def __init__(self, image_dir, n=1, transform=None):
        self.image_dir = image_dir
        self.n = n
        self.transform = transform
        self.image_paths = []
        self.folder_to_images = {}

        for root, _, files in os.walk(image_dir):
            files = sorted([f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
            full_paths = [os.path.join(root, f) for f in files]

            if full_paths:
                self.folder_to_images[root] = full_paths
                self.image_paths.extend(full_paths)  
    def __len__(self):
        return len(self.image_paths)

    

    def __getitem__(self, idx):
        target_path = self.image_paths[idx]
        folder = os.path.dirname(target_path)
        image_list = self.folder_to_images[folder]
        pos_in_folder = image_list.index(target_path)

        start = max(0, pos_in_folder - self.n)
        end = min(len(image_list), pos_in_folder + self.n + 1)
        selected = image_list[start:end]

        while len(selected) < 2 * self.n + 1:
            if start == 0:
                selected.insert(0, selected[0]) 
            else:
                selected.append(selected[-1]) 

        images = [Image.open(p).convert('L') for p in selected]

        if self.transform:
            images = [self.transform(img) for img in images]

        images = torch.cat(images, dim=0)

        return images, target_path



class realImageDataset(Dataset):
    def __init__(self, root_dir, size = (512,512), neibor = 1):
        self.root_dir = root_dir
        self.image_paths = []
        self.size = size
        self.neibor = neibor
        # Collect image file paths recursively
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))

        # Sort image paths to maintain consistent order
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        

        if self.neibor > 1:
            # Multiply input
            target_forder = os.path.basename(img_path).split('+')[0]
            target_name = os.path.basename(img_path).split('+')[1]
            
            base_name, ext = os.path.splitext(target_name)
            base_num = int(base_name.split('_')[0])
            
            images = []
            for i in range(-int(self.neibor / 2), int(self.neibor / 2) + 1):
                neighbor_num = base_num + i
                neighbor_name = f"{neighbor_num:04d}" + f"{base_name[4:]}{ext}"
                file_path = Path(img_path)
                target_dir = file_path.parent.parent  # 两次 `.parent` 回溯两级

                neighbor_path = os.path.join(target_dir, 'OriAll', target_forder, neighbor_name)

                if os.path.exists(neighbor_path):
                    neighbor_image = Image.open(neighbor_path).convert("L")
                else:
                    # print(f"File not found: {neighbor_path}, using last image.")
                    neighbor_image = images[-1] if images else Image.open(img_path).convert("L")  

                images.append(neighbor_image)   
        else:
            image = Image.open(img_path).convert("L")
            images = [image]

        deVesselPath = os.path.join(os.path.dirname(img_path) + '_deVessel_forBack', os.path.basename(img_path))
        deVesselImg = Image.open(deVesselPath).convert("L") 

        deVesselPath_vessel = os.path.join(os.path.dirname(img_path) + '_deVessel_forVessel', os.path.basename(img_path))
        deVesselImg_vessel = Image.open(deVesselPath_vessel).convert("L") 

        deVesselFillPath = os.path.join(os.path.dirname(img_path) + '_deVesselFill_forVessel', os.path.basename(img_path))
        deVesselFillImg = Image.open(deVesselFillPath).convert("L") 

        # deVesselFillPath_real = os.path.join(os.path.dirname(img_path) + '_deVesselFill_forVessel_real', os.path.basename(img_path))
        # deVesselFillImg_real = Image.open(deVesselFillPath_real).convert("L") 

        transformed_images = []
        random_p = random.random()
        for image in images:
            if isinstance(image, Image.Image):
                image = F.center_crop(image, self.size)
                if random_p > 0.5:
                    image = F.hflip(image)
                image = F.to_tensor(image)
            transformed_images.append(image)

        # image = F.center_crop(image,self.size)
        deVesselImg = F.center_crop(deVesselImg,self.size)
        deVesselImg_vessel = F.center_crop(deVesselImg_vessel,self.size)
        deVesselFillImg = F.center_crop(deVesselFillImg,self.size)
        # deVesselFillImg_real = F.center_crop(deVesselFillImg_real,self.size)


        if random_p > 0.5:
            # image = F.hflip(image)
            deVesselImg = F.hflip(deVesselImg)
            deVesselImg_vessel = F.hflip(deVesselImg_vessel)
            deVesselFillImg = F.hflip(deVesselFillImg)
            # deVesselFillImg_real = F.hflip(deVesselFillImg_real)

        # image = F.to_tensor(image) 
        deVesselImg = F.to_tensor(deVesselImg)
        deVesselImg_vessel = F.to_tensor(deVesselImg_vessel)
        deVesselFillImg = F.to_tensor(deVesselFillImg)
        # deVesselFillImg_real = F.to_tensor(deVesselFillImg_real)

        return {
            "image": torch.cat(transformed_images, dim=0),
            "image_deVessel": deVesselImg,
            "image_deVessel_vessel": deVesselImg_vessel,
            "image_filled": deVesselFillImg,
            # "image_filled_real": deVesselFillImg_real
        }

    
class PairedDataset(Dataset):
    def __init__(self, path,  size=(512,512), transform=None,channel=1):
        self.image1_dir = os.path.join(path,'image1')
        self.image2_dir = os.path.join(path,'image2')
        self.label1_dir = os.path.join(path,'label')
        self.label2_dir = os.path.join(path,'label2')
        self.image1_dir_ori = os.path.join(path,'image1_ori')
        self.image2_dir_ori = os.path.join(path,'image2_ori')
        self.label_dir_ori = os.path.join(path,'label_ori')
        self.label_dir_ori2 = os.path.join(path,'label2_ori')
        self.image_names = os.listdir(self.image1_dir)
        self.transform = transform
        self.size = size
        self.channel = channel

        # assert set(self.image_names) == set(os.listdir(self.image2_dir)) == set(os.listdir(self.label_dir)), \
        #     "File names in the directories do not match!"

    def __len__(self):
        return len(self.image_names)

    def apply_padding(self,img, pad):
        if img is None:
            return None
        return F.pad(img, padding=pad, fill=0)  # black padding

    def resize_if_needed(self, img):
        if img != 0:
            w, h = img.size
            target_h, target_w = self.size
            if h < target_h or w < target_w:
                return F.resize(img, self.size)
        return img
    
    def generate_elastic_transform(self, image_shape, alpha=1.0, sigma=10.0):
        shape = image_shape
        random_state = np.random.RandomState(None)
        
        dx = random_state.uniform(-1, 1, shape) * alpha
        dy = random_state.uniform(-1, 1, shape) * alpha

        dx = gaussian_filter(dx, sigma, mode="constant", cval=0)
        dy = gaussian_filter(dy, sigma, mode="constant", cval=0)

        return dx, dy

    def apply_elastic_transform(self, image, alpha=1.0, sigma=10.0):

        shape = image.shape
        dx, dy = self.generate_elastic_transform(shape, alpha, sigma)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        distorted_x = np.clip(x + dx, 0, shape[1] - 1)
        distorted_y = np.clip(y + dy, 0, shape[0] - 1)
        distorted_image = map_coordinates(image, [distorted_y, distorted_x], order=1, mode='reflect')
        return distorted_image
    

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Define all paths
        paths = {
            'image1': os.path.join(self.image1_dir, image_name),
            'image2': os.path.join(self.image2_dir, image_name),
            'label1': os.path.join(self.label1_dir, image_name),
            'label2': os.path.join(self.label2_dir, image_name),
            'image1_ori': os.path.join(self.image1_dir_ori, image_name),
            'image2_ori': os.path.join(self.image2_dir_ori, image_name),
            'label1_ori': os.path.join(self.label_dir_ori, image_name),
            'label2_ori': os.path.join(self.label_dir_ori2, image_name),
        }

        #Load images
        data = {}
        for key, path in paths.items():
            data[key] = Image.open(path).convert('L') if os.path.exists(path) else 0

        # Resize 
        for key in data:
            data[key] = self.resize_if_needed(data[key])

        if self.transform:
            # --- Step 4: Unified random padding ---
            pad = random.randint(5, 10)
            for key in data:
                data[key] = self.apply_padding(data[key], pad)

            # Consistent random crop
            if data['image1'] != 0 and data['image2'] != 0 and data['label1'] != 0:
                i, j, h, w = transforms.RandomCrop.get_params(data['image1'], output_size=self.size)
                for key in data:
                    if data[key] != 0:
                        data[key] = F.crop(data[key], i, j, h, w)

                # Consistent random horizontal flip 
                if random.random() > 0.5:
                    for key in data:
                        if data[key] != 0:
                            data[key] = F.hflip(data[key])

            for key in data:
                if data[key] != 0:
                    data[key] = F.to_tensor(data[key])
        else:
            for key in data:
                if data[key] != 0:
                    data[key] = F.center_crop(data[key], self.size)
                    data[key] = F.to_tensor(data[key])
        
        if self.channel > 1 and self.transform:
            non_blur_funcs = [
                lambda x: F.adjust_gamma(x, random.uniform(0.8, 1.3)),
                lambda x: F.gaussian_blur(x, kernel_size=5, sigma=1)
                
            ]
            def blur_then_erase(x):
                x = RandomErasing(p=1, scale=(0.1, 0.33), ratio=(0.3, 3.3), value=0.48)(x)
                x = F.gaussian_blur(x, kernel_size=3, sigma=1)
                return x
            blur_func = lambda x: blur_then_erase(x)

            selected_funcs = []
            for ch in range(self.channel):
                if random.random() < 0.5:
                    selected_funcs.append((blur_func, True))
                else:
                    selected_funcs.append((random.choice(non_blur_funcs), False))
            if all(is_blur for (_, is_blur) in selected_funcs):
                selected_funcs[self.channel//2] = (random.choice(non_blur_funcs), False)

            mask1 = (data['label1'] > 0.5).float()
            mask2 = (data['label2'] > 0.5).float()

            enhanced_channels1 = []
            enhanced_channels2 = []

            selected_funcs = []
            for ch in range(self.channel):
                if random.random() < 0.5:
                    selected_funcs.append((blur_func, True))
                else:
                    selected_funcs.append((random.choice(non_blur_funcs), False))

            mid_ch = self.channel // 2
            if all(is_blur for (_, is_blur) in selected_funcs):
                selected_funcs[mid_ch] = (random.choice(non_blur_funcs), False)

            for ch, (func, is_blur) in enumerate(selected_funcs):
                enhanced_img1 = func(data['image1'])
                enhanced_img2 = func(data['image2'])

                channel_img1 = data['image1'] * (1 - mask1) + enhanced_img1 * mask1
                channel_img2 = data['image2'] * (1 - mask2) + enhanced_img2 * mask2

                if ch != mid_ch:
                    # Tensor -> numpy for elastic transform
                    np_img1 = channel_img1.squeeze(0).numpy()
                    np_img2 = channel_img2.squeeze(0).numpy()

                    alpha = np.random.uniform(5.0, 100.0)
                    sigma = np.random.uniform(5.0, 15.0)
                    elastic1 = self.apply_elastic_transform(np_img1,alpha=alpha, sigma=sigma)
                    elastic2 = self.apply_elastic_transform(np_img2,alpha=alpha, sigma=sigma)

                    # numpy -> tensor with 1 channel
                    channel_img1 = torch.from_numpy(elastic1).unsqueeze(0).float()
                    channel_img2 = torch.from_numpy(elastic2).unsqueeze(0).float()

                enhanced_channels1.append(channel_img1)
                enhanced_channels2.append(channel_img2)

            data['image1'] = torch.cat(enhanced_channels1, dim=0)
            data['image2'] = torch.cat(enhanced_channels2, dim=0)

        return data


