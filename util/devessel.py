# import os
# import shutil

# def copy_selected_images(source_folder, target_folder):
#     # Ensure the target folder exists
#     os.makedirs(target_folder, exist_ok=True)

#     # Iterate through each subfolder in the source directory
#     for j, subfolder in enumerate(os.listdir(source_folder)):
#         subfolder_path = os.path.join(source_folder, subfolder)

#         if os.path.isdir(subfolder_path):
#             images = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
#             # Check if the number of images is greater than 20
#             if len(images) > 20:
#                 for i in range(14, len(images), 8):  # Start from index 14 (15th image), step by 5
#                     image_path = os.path.join(subfolder_path, images[i])
#                     target_path = os.path.join(target_folder, str(j) + '_' + images[i])
#                     shutil.copy(image_path, target_path)
#                     print(f"Copied: {image_path} to {target_path}")

# if __name__ == "__main__":
#     source_folder = "data/Private/OriAll"
#     target_folder = "data/Private/ImageWithVessel"
#     copy_selected_images(source_folder, target_folder)

import os
import shutil
import cv2 
import numpy as np
import matlab.engine

def copy_selected_images(source_folder, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 遍历源文件夹中的每个子文件夹
    for j, subfolder in enumerate(os.listdir(source_folder)):
        subfolder_path = os.path.join(source_folder, subfolder)

        if os.path.isdir(subfolder_path):
            # 获取并排序所有图片文件
            images = sorted([f for f in os.listdir(subfolder_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # 仅在图片数量大于20时处理
            if len(images) > 20:
                # 计算中间索引
                mid_index = len(images) // 2
                
                # 获取中间图像
                middle_image = images[mid_index]
                
                # 构建路径
                image_path = os.path.join(subfolder_path, middle_image)
                target_path = os.path.join(target_folder, f"{j}_{middle_image}")
                
                # 复制文件
                shutil.copy(image_path, target_path)
                print(f"Copied: {image_path} to {target_path}")

def add_label(img, label, label_height=40):
    """
    在图像上方添加一块区域用于显示标签文字，并返回带标签的新图像。
    :param img: 原始图像（灰度）
    :param label: 要显示的文字标签（字符串）
    :param label_height: 标签区域高度，单位像素
    :return: 带有标签区域的新图像
    """
    h, w = img.shape[:2]
    # 新图像高度 = 标签高度 + 原图高度
    new_img = np.zeros((h + label_height, w), dtype=img.dtype)
    # 将原图复制到新图像的下方区域
    new_img[label_height:, :] = img

    # 定义文字参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    # 获取文字尺寸
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size

    # 计算文字放置位置（居中）
    x = (w - text_width) // 2
    y = (label_height + text_height) // 2

    # 在标签区域内添加文字（使用白色文字）
    cv2.putText(new_img, label, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
    return new_img

def process_images(source_folder, target_folder, thresholds=[80, 100, 120, 150]):
    """
    遍历 source_folder 下所有子文件夹中的图像，
    对每张图像分别使用给定的阈值进行处理（将像素值低于阈值的置为 0），
    并在每个图像上方添加文字标注（原图标注为“原图”，处理后图像标注为“阈值XX”），
    将原图和各个阈值处理后的图像水平拼接保存到 target_folder 中，
    同时保持原有的文件夹结构。
    """
    # 支持的图像后缀
    img_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for dirpath, dirnames, filenames in os.walk(source_folder):
        for filename in filenames:
            if not filename.lower().endswith(img_extensions):
                continue

            img_path = os.path.join(dirpath, filename)
            # 以灰度模式读取图像
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法读取图像：{img_path}")
                continue

            annotated_images = []

            # 原图添加标签“原图”
            annotated_orig = add_label(img, "Original")
            annotated_images.append(annotated_orig)

            # 分别对图像使用不同阈值处理并添加对应标签
            for t in thresholds:
                # 使用 cv2.THRESH_TOZERO：低于阈值的置为 0，高于阈值的不变
                _, thresh_img = cv2.threshold(img, t, 255, cv2.THRESH_TOZERO)
                annotated_thresh = add_label(thresh_img, f"Threshold {t}")
                annotated_images.append(annotated_thresh)

            # 水平拼接所有标注后的图像
            concatenated = cv2.hconcat(annotated_images)

            # 构造目标保存路径，保持原有的文件夹结构
            relative_path = os.path.relpath(dirpath, source_folder)
            target_dir = os.path.join(target_folder, relative_path)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, filename)

            cv2.imwrite(target_path, concatenated)
            print(f"已处理并保存：{target_path}")


def process_images_with_matlab(source_folder, target_folder,target_fill_folder,target_folder_cmb):
    """
    遍历 source_folder 下的所有子文件夹中的图像，
    对每张图像调用 MATLAB 函数 deVessel 进行处理，
    然后将原图和 MATLAB 处理后的图像水平拼接，
    并按照原有的文件夹结构保存在 target_folder 下。
    """
    # 支持的图像后缀
    img_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # 启动 MATLAB 引擎
    print("启动 MATLAB 引擎...")
    eng = matlab.engine.start_matlab()

    # 遍历所有子文件夹及图像
    for dirpath, _, filenames in os.walk(source_folder):
        for filename in filenames:
            if not filename.lower().endswith(img_extensions):
                continue

            img_path = os.path.join(dirpath, filename)
            # 以灰度模式读取图像（若需要处理彩色图像请修改读取方式）
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法读取图像：{img_path}")
                continue

            print(f"处理图像：{img_path}")

            # 将 numpy 数组转换为 MATLAB 数据类型
            # matlab.uint8() 需要传入一个二维列表
            img_list = img.tolist()
            matlab_img = matlab.uint8(img_list)

            # 调用 MATLAB 函数 deVessel 进行处理
            # 假设 deVessel 函数的输入为 uint8 类型图像，返回处理后的图像矩阵
            try:
                processed, filled_image = eng.deVessel(matlab_img, nargout=2)
            except Exception as e:
                print(f"调用 deVessel 出错：{e}")
                continue

            # 将 MATLAB 返回的数据转换为 numpy 数组
            # 注意：转换后数据类型可能为 float，需转换回 uint8
            processed_np = np.array(processed)
            # 若数据范围不在 0-255 内，可根据实际情况进行归一化处理
            processed_np = processed_np.astype(np.uint8)

            filled_image_np = np.array(filled_image)
            # 若数据范围不在 0-255 内，可根据实际情况进行归一化处理
            filled_image_np = filled_image_np.astype(np.uint8)

            # # 为确保拼接时图像尺寸一致，如高度不匹配则进行调整
            # h1, w1 = img.shape[:2]
            # h2, w2 = processed_np.shape[:2]
            # if h1 != h2:
            #     processed_np = cv2.resize(processed_np, (w2, h1))

            # 水平拼接原图和处理后的图像
            concatenated = cv2.hconcat([img, processed_np,filled_image_np])

            # 构造目标保存路径，保持原有的文件夹结构
            relative_path = os.path.relpath(dirpath, source_folder)

            target_dir = os.path.join(target_folder, relative_path)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, filename)
            cv2.imwrite(target_path, processed_np)

            target_fill_dir = os.path.join(target_fill_folder, relative_path)
            os.makedirs(target_fill_folder, exist_ok=True)
            target_fill_path = os.path.join(target_fill_dir, filename)
            cv2.imwrite(target_fill_path, filled_image_np)

            target_cmb_dir = os.path.join(target_folder_cmb, relative_path)
            os.makedirs(target_cmb_dir, exist_ok=True)
            target_cmb_path = os.path.join(target_cmb_dir, filename)
            cv2.imwrite(target_cmb_path, concatenated)
            print(f"已保存拼接图像：{target_path}")

    # 关闭 MATLAB 引擎
    eng.quit()
    print("全部处理完成！")

if __name__ == '__main__':
    source_folder = "/mydata/myProject/21.CAS/output/Combined4/sameBackground_augmentedVessel_strong2/realImages-2"  # 源文件夹路径
    target_folder = "/mydata/myProject/21.CAS/output/Combined4/Devessel/realImages_deVessel"               # 目标文件夹路径
    target_folder_fill = "/mydata/myProject/21.CAS/output/Combined4/Devessel/realImages_deVesselFill"
    target_folder_cmb = "/mydata/myProject/21.CAS/output/Combined4/Devessel/realImages_deVesselCMB"
    # process_images(source_folder, target_folder)
    process_images_with_matlab(source_folder, target_folder,target_folder_fill,target_folder_cmb)

# if __name__ == "__main__":
#     source_folder = "data/Private/OriAll"
#     # target_folder = "data/Private/vesselImage"
#     copy_selected_images(source_folder, target_folder)