import os
import pydicom
import numpy as np
import cv2

def save_slices_from_single_dicom(dicom_file, output_folder):
    """
    将单个 DICOM 文件中的切片保存为 PNG 文件。
    
    :param dicom_file: DICOM 文件路径。
    :param output_folder: 保存 PNG 文件的文件夹路径。
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取 DICOM 文件
    ds = pydicom.dcmread(dicom_file)

    # 检查是否有多个切片
    if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
        num_slices = ds.NumberOfFrames
        # print(f"Processing {num_slices} slices from the DICOM file.")
        
        # 逐帧提取像素数据
        for idx in range(num_slices):
            if len(ds.pixel_array[idx].shape)<2:
                 continue
            image = ds.pixel_array[idx]

            if np.max(image)>255:
                print('max value of image is:',num_slices, os.path.basename(dicom_file),np.min(image), np.max(image))
            else:
                 print('255 value, number slice:',num_slices)
            # 创建新的 DICOM 文件
            if os.path.basename(dicom_file) == 'XA.1.3.12.2.1107.5.4.5.109155.30000022080412365960400000062':
                 print(os.path.basename(dicom_file))
            single_frame_dicom = ds.copy()  # 复制原始 DICOM 的元数据
            single_frame_dicom.PixelData = image.tobytes()  # 替换为单帧数据
            
            single_frame_dicom.Rows, single_frame_dicom.Columns = image.shape  # 更新图像维度
            single_frame_dicom.NumberOfFrames = 1  # 更新帧数信息

            # 生成输出路径并保存
            output_path_dcm = os.path.join(output_folder,'dcm', f"{idx:04d}_0000.dcm")
            os.makedirs(os.path.dirname(output_path_dcm), exist_ok=True)
            single_frame_dicom.save_as(output_path_dcm)

            #npy文件
            output_path_npy = os.path.join(output_folder,'npy', f"{idx:04d}_0000.npy")
            os.makedirs(os.path.dirname(output_path_npy), exist_ok=True)
            np.save(output_path_npy,image)
            # print(f"Saved slice {idx} to {output_path}")
    else:
        # 单帧 DICOM 文件
        idx=0
        # print("Single-frame DICOM file detected.")
        image = ds.pixel_array

        # 创建新的 DICOM 文件
        single_frame_dicom = ds.copy()  # 复制原始 DICOM 的元数据
        single_frame_dicom.PixelData = image.tobytes()  # 替换为单帧数据
        single_frame_dicom.Rows, single_frame_dicom.Columns = image.shape  # 更新图像维度
        single_frame_dicom.NumberOfFrames = 1  # 更新帧数信息

        # 生成输出路径并保存
        output_path_dcm = os.path.join(output_folder,'dcm', f"{idx:04d}_0000.dcm")
        os.makedirs(os.path.dirname(output_path_dcm), exist_ok=True)
        single_frame_dicom.save_as(output_path_dcm)

        #npy文件
        output_path_npy = os.path.join(output_folder,'npy', f"{idx:04d}_0000.npy")
        os.makedirs(os.path.dirname(output_path_npy), exist_ok=True)
        np.save(output_path_npy,image)

# 示例用法
# dicom_file = "data/XA.1.3.46.670589.28.681722603677250.202208202005523056702211511"  # 替换为你的 DICOM 文件路径
# output_folder = "data/savedDSA"  # 替换为保存 PNG 的文件夹路径
# save_slices_from_single_dicom(dicom_file, output_folder)
root_folder = 'data/Private/CardiacCathsForSyed'
output_root_folder = 'data/Private/OriAll_DCM'
for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.startswith("XA."):
                dicom_file_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(dirpath, root_folder)
                output_folder = os.path.join(output_root_folder, filename)
                
                # print(f"Processing {dicom_file_path}...")
                save_slices_from_single_dicom(dicom_file_path, output_folder)
