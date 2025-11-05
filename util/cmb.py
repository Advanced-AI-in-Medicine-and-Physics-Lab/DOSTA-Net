import cv2
import numpy as np
import random
from itertools import islice, cycle
import os
import SimpleITK as sitk

def random_contrast_gamma_transform(img, contrast_range=(0.87, 1.13), gamma_range=(0.87, 1.13)):
    """
    Apply random contrast and gamma transformations to a grayscale image.
    """
    # Randomly select contrast and gamma values within the given ranges
    contrast_factor = random.uniform(*contrast_range)
    gamma_value = random.uniform(*gamma_range)
    
    # Contrast adjustment: I_new = α * (I - mean) + mean
    mean_intensity = np.mean(img)
    contrast_adjusted = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=(1 - contrast_factor) * mean_intensity)
    
    # Gamma adjustment: I_new = ((I / 255) ^ gamma) * 255
    img_normalized = contrast_adjusted / 255.0
    gamma_corrected = np.power(img_normalized, gamma_value) * 255
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    
    return gamma_corrected


# def transfromVessel(image_b):
#     # Randomly scale image B
#     scale_factor = random.uniform(0.85, 0.99)
#     new_size = (int(image_b.shape[1] * scale_factor), int(image_b.shape[0] * scale_factor))
#     scaled_b = cv2.resize(image_b, new_size)

#     # Create a blank canvas with the original size
#     padded_b = np.zeros_like(image_b)

#     # Calculate maximum offsets to ensure the scaled image stays within bounds
#     max_y_offset = padded_b.shape[0] - scaled_b.shape[0]
#     max_x_offset = padded_b.shape[1] - scaled_b.shape[1]

#     # Randomly calculate offsets within the valid range
#     y_offset = random.randint(0, max_y_offset)
#     x_offset = random.randint(0, max_x_offset)

#     # Place the scaled image on the blank canvas
#     padded_b[y_offset:y_offset + scaled_b.shape[0], x_offset:x_offset + scaled_b.shape[1]] = scaled_b

#     # Create a mask for dark regions in the padded image B
#     # _, mask = cv2.threshold(padded_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     _, mask = cv2.threshold(padded_b, 20, 255, cv2.THRESH_BINARY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Highlight dark regions of padded B in color
#     # reversed_b = 255 - padded_b

#     return padded_b, mask

# def transfromVessel(image_b):
#     # Randomly scale image B
#     scale_factor = random.uniform(0.85, 0.99)
#     new_size = (int(image_b.shape[1] * scale_factor), int(image_b.shape[0] * scale_factor))
#     scaled_b = cv2.resize(image_b, new_size)

#     # Create a blank canvas with the original size
#     padded_b = np.zeros_like(image_b)

#     # Calculate maximum offsets to ensure the scaled image stays within bounds
#     max_y_offset = padded_b.shape[0] - scaled_b.shape[0]
#     max_x_offset = padded_b.shape[1] - scaled_b.shape[1]

#     # Randomly calculate offsets within the valid range
#     y_offset = random.randint(0, max_y_offset)
#     x_offset = random.randint(0, max_x_offset)

#     # Place the scaled image on the blank canvas
#     padded_b[y_offset:y_offset + scaled_b.shape[0], x_offset:x_offset + scaled_b.shape[1]] = scaled_b

#     # Create a mask for dark regions in the padded image B
#     _, mask = cv2.threshold(padded_b, 20, 255, cv2.THRESH_BINARY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Randomly rotate the image and mask (0, 90, 180, 270 degrees)
#     angle = random.choice([0, 90, 180, 270])
#     if angle > 0:
#         rotated_b = cv2.rotate(padded_b, {90: cv2.ROTATE_90_CLOCKWISE, 
#                                           180: cv2.ROTATE_180, 
#                                           270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
#         rotated_mask = cv2.rotate(mask, {90: cv2.ROTATE_90_CLOCKWISE, 
#                                          180: cv2.ROTATE_180, 
#                                          270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
#     else:
#         rotated_b, rotated_mask = padded_b, mask

#     return rotated_b, rotated_mask

import cv2
import random
import numpy as np

def add_quantum_noise(image, dose_range=(0.2, 0.8)):
    """添加X射线量子噪声"""
    dose_factor = random.uniform(*dose_range)
    max_val = image.max()
    scaled = image * dose_factor
    noisy = np.random.poisson(scaled) / dose_factor
    return np.clip(noisy, 0, max_val).astype(image.dtype)

def add_structural_noise(image, mask):
    """添加血管结构相关噪声"""
    # 血管边缘模糊
    if random.random() > 0.2:
        kernel_size = random.choice([3,5,7])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # 生成血流伪影
    if random.random() > 0.2:
        direction = random.choice(['h', 'v'])
        kernel = np.zeros((15,15))
        center = 7
        if direction == 'h':
            kernel[center,:] = 1
        else:
            kernel[:,center] = 1
        kernel /= kernel.sum()
        streak = cv2.filter2D(mask, -1, kernel)
        image = cv2.addWeighted(image, 0.8, streak, 0.2, 0)
    
    return image

def add_sensor_noise(image):
    """添加传感器相关噪声"""
    # 高斯电子噪声
    sigma = random.uniform(0.02, 0.08)
    gaussian = np.random.normal(0, sigma, image.shape)
    
    # PRNU像素响应差异
    prnu = 1 + np.random.normal(0, 0.03, image.shape)
    
    # 暗电流噪声
    dark = np.random.uniform(0, 0.05, image.shape)
    
    noisy = image * prnu + gaussian*255 + dark*255
    return np.clip(noisy, 0, 255).astype(image.dtype)

def transfromVessel(image_b):
    # 原始几何变换流程
    scale_factor = random.uniform(0.7, 0.99)
    new_size = (int(image_b.shape[1] * scale_factor), 
               int(image_b.shape[0] * scale_factor))
    scaled_b = cv2.resize(image_b, new_size)
    
    padded_b = np.zeros_like(image_b)
    max_y = padded_b.shape[0] - scaled_b.shape[0]
    max_x = padded_b.shape[1] - scaled_b.shape[1]
    y_offset = random.randint(0, max_y)
    x_offset = random.randint(0, max_x)
    padded_b[y_offset:y_offset+scaled_b.shape[0], 
             x_offset:x_offset+scaled_b.shape[1]] = scaled_b

    # 生成初始掩膜
    _, mask = cv2.threshold(padded_b, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 随机旋转
    angle = random.choice([0, 90, 180, 270])
    if angle > 0:
        rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }[angle]
        rotated_b = cv2.rotate(padded_b, rotate_code)
        rotated_mask = cv2.rotate(mask, rotate_code)
    else:
        rotated_b, rotated_mask = padded_b, mask
    
    # ==== 新增噪声处理流程 ====
    # 步骤1：添加解剖结构相关噪声
    rotated_b_struct = add_structural_noise(rotated_b, rotated_mask)
    
    # 步骤2：添加量子噪声（需要转换为浮点型）
    rotated_b_float = rotated_b_struct.astype(np.float32)
    rotated_b_float = add_quantum_noise(rotated_b_float)
    
    # 步骤3：添加传感器噪声
    rotated_b_noisy = add_sensor_noise(rotated_b_float)
    
    # 步骤4：组织纹理噪声
    if random.random() > 0.2:
        texture = cv2.GaussianBlur(np.random.randn(*rotated_b.shape), (51,51), 0)
        rotated_b_noisy = rotated_b_noisy * 0.9 + texture * 0.1 * 255
    
    # 最终裁剪和类型转换
    final_image = np.clip(rotated_b_noisy, 0, 255).astype(np.uint8)
    
    return final_image, rotated_mask,rotated_b


def normlized_img(arr):
    # 归一化到 [0, 1]
    arr_min = arr.min()
    arr_max = arr.max()
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    normalized_arr = normalized_arr*255
    return normalized_arr.astype(np.uint8)

def random_process_image(image):
    """
    Randomly resize the image to (300,300) or randomly crop a (300,300) region.

    Parameters:
        image (numpy.ndarray): Input image to be processed.

    Returns:
        numpy.ndarray: Processed image with size (300,300).
    """
    choice = random.choice(["resize", "crop"])

    if choice == "resize":
        processed_image = cv2.resize(image, (300, 300))
    else:
        h, w = image.shape[:2]
        if h < 300 or w < 300:
            # If the image is smaller than the crop size, resize it first
            image = cv2.resize(image, (max(w, 300), max(h, 300)))
            h, w = image.shape[:2]
        
        # Randomly select top-left corner for cropping
        x_start = random.randint(0, w - 300)
        y_start = random.randint(0, h - 300)
        processed_image = image[y_start:y_start + 300, x_start:x_start + 300]

    return processed_image
    
def fuse_images_with_scaling_sameBackground(image_a_path, image_b_path, image_b2_path, output_path, num, threshold=50):
    """
    Fuse two images by resizing and scaling image B, then blending it with image A.

    Parameters:
        image_a_path (str): Path to the image A (background image).
        image_b_path (str): Path to the image B (highlight dark areas).
        output_path (str): Path to save the fused image.
        weight (float): Weight for blending image A with image B's highlights (default: 0.7).
        threshold (int): Threshold to detect dark regions in image B (default: 50).
    Returns:
        None
    """

    # Load images
    image_a = cv2.imread(image_a_path, cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread(image_b_path, cv2.IMREAD_GRAYSCALE)
    image_b2 = cv2.imread(image_b2_path, cv2.IMREAD_GRAYSCALE)

    image_a = random_process_image(image_a)
    #Diversify background 
    image_a = random_contrast_gamma_transform(image_a)

    # Resize image B to match image A's dimensions
    image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]))

    # Resize image B to match image A's dimensions
    image_b2 = cv2.resize(image_b2, (image_a.shape[1], image_a.shape[0]))

    #diversify vessel image by resize
    pad_b,mask_b, noNoise_b = transfromVessel(image_b)
    pad_b2,mask_b2,noNoise_b2 = transfromVessel(image_b2)

    image_array = image_b2.astype(np.float32)  - pad_b.astype(np.float32) 
    image = sitk.GetImageFromArray(image_array)
    
    saveName_n = str(num).zfill(5)+'.nii.gz'
    save_nii = os.path.join(output_path, 'nii', saveName_n)
    os.makedirs(os.path.dirname(save_nii),exist_ok=True)

    sitk.WriteImage(image, 'output_2d_negative.nii.gz')


    # Blend image A and the highlighted regions
    # fused_imageb1 = cv2.addWeighted(image_a, 1, -pad_b, 1, 0)
    # fused_imageb2 = cv2.addWeighted(image_a, 1, -pad_b2, 1, 0)
    ratio = random.uniform(0.25, 0.34)
    fused_imageb1 = image_a.astype(np.int16) - (pad_b*ratio).astype(np.int16)
    fused_imageb2 = image_a.astype(np.int16) - (pad_b2*ratio).astype(np.int16)

    fused_imageb1 = np.clip(fused_imageb1, 0, None)
    fused_imageb2 = np.clip(fused_imageb2, 0, None)
   
    # Save the result
    saveName = str(num).zfill(5)+'.png'
    save_image1 = os.path.join(output_path, 'image1', saveName)
    save_image2 = os.path.join(output_path, 'image2', saveName)
    save_image1_ori = os.path.join(output_path, 'image1_ori', saveName)
    save_image2_ori = os.path.join(output_path, 'image2_ori', saveName)
    save_label = os.path.join(output_path, 'label', saveName)
    save_label_ori = os.path.join(output_path, 'label_ori', saveName)
    save_label2 = os.path.join(output_path, 'label2', saveName)
    save_label2_ori = os.path.join(output_path, 'label2_ori', saveName)

    os.makedirs(os.path.dirname(save_image1),exist_ok=True)
    os.makedirs(os.path.dirname(save_image2),exist_ok=True)
    os.makedirs(os.path.dirname(save_image1_ori),exist_ok=True)
    os.makedirs(os.path.dirname(save_image2_ori),exist_ok=True)
    os.makedirs(os.path.dirname(save_label),exist_ok=True)
    os.makedirs(os.path.dirname(save_label_ori),exist_ok=True)
    os.makedirs(os.path.dirname(save_label2),exist_ok=True)
    os.makedirs(os.path.dirname(save_label2_ori),exist_ok=True)

    cv2.imwrite(save_image1, fused_imageb1)
    cv2.imwrite(save_image2, fused_imageb2)
    cv2.imwrite(save_label, mask_b)
    cv2.imwrite(save_label2, mask_b2)

    cv2.imwrite(save_image1_ori, image_a)
    cv2.imwrite(save_image2_ori, image_a)
    cv2.imwrite(save_label_ori, (noNoise_b*ratio).astype(np.uint8))
    cv2.imwrite(save_label2_ori, (noNoise_b2*ratio).astype(np.uint8))

    save_cmb = os.path.join(output_path, 'CMB', saveName)
    os.makedirs(os.path.dirname(save_cmb),exist_ok=True)

    # stacked_image = np.hstack((image_a, fused_imageb1, fused_imageb2, (255-pad_b).astype(np.uint8), (255-pad_b2).astype(np.uint8)))
    stacked_image = np.hstack((image_a, pad_b.astype(np.uint8), fused_imageb1,  (255-pad_b).astype(np.uint8)))
    cv2.imwrite(save_cmb, stacked_image)


def process_image(image, fused_image):

    image_3channel = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    fused_image3 = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2RGB)

    image_3channel = image_3channel.transpose((2, 0, 1))
    fused_image3 = fused_image3.transpose((2, 0, 1))
    # Process the 3-channel image
    processed_image = FDA_source_to_target_np(image_3channel, fused_image3, L=0.05)

    # 转置回 (H, W, C) 格式
    src_in_trg = processed_image.transpose((1, 2, 0))

    # 剪裁并归一化
    cmin, cmax = 0.0, 255.0
    normalized = np.clip(src_in_trg, cmin, cmax)
    normalized = ((normalized - cmin) / (cmax - cmin) * 255).astype(np.uint8)
       
    return np.mean(normalized, axis=-1).astype(image.dtype)

# Example usage

from tqdm import tqdm
# Example usage
path_vessel = '/mydata/myProject/21.CAS/synthetic-ddsa/outputProject/CA4'
path_frame = 'data/Private/firstFrame_clean_deleteBad'
savePath = '/mydata/myProject/21.CAS/output/Combined4/sameBackground_augmentedVessel_strong2/Train'

listFrame = os.listdir(path_frame)
listVessel = os.listdir(path_vessel)

listVessel = sorted(listVessel, key=lambda x: random.random())


listFrame = list(islice(cycle(listFrame), len(listVessel)))

# for i, vessel in enumerate(tqdm(listVessel)):
#     vessel_p = os.path.join(path_vessel, vessel)

#     image1_p = os.path.join(path_frame, listFrame[i])
#     index2 = i + random.randint(10, 20)
#     if index2 > len(listVessel)-1:
#         index2 = i - random.randint(20, 30)
#     image2_p = os.path.join(path_frame, listFrame[index2])

#     fuse_images_with_scaling_sameVessel(image1_p, image2_p, vessel_p,savePath,i, threshold=10)


for i, vessel in enumerate(tqdm(listVessel)):
    vessel_p = os.path.join(path_vessel, vessel)

    image1_p = os.path.join(path_frame, listFrame[i])

    index2 = i + random.randint(10, 20)
    if index2 > len(listVessel)-1:
        index2 = i - random.randint(20, 30)
    vessel_p2 = os.path.join(path_vessel, listVessel[index2])

    fuse_images_with_scaling_sameBackground(image1_p, vessel_p, vessel_p2, savePath,i, threshold=10)