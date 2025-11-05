% function final_image = process_image(path,mean_value)
% I=double(imread(path));
% options = struct('FrangiScaleRange', [1 8], 'FrangiScaleRatio', 1, 'FrangiBetaOne', 2, 'FrangiBetaTwo', 8, 'verbose',true,'BlackWhite',true);
% 
% Ivessel=FrangiFilter2D(I,options);
% %figure,
% %subplot(1,2,1), imshow(I,[]);
% %subplot(1,2,2), imshow(Ivessel,[0 0.25]);
% 
% % 1. 图像二值化
% threshold = graythresh(Ivessel); % 自动确定阈值
% binary_image = imbinarize(Ivessel, threshold);
% 
% % 2. 进行闭操作（先膨胀再腐蚀）
% se = strel('disk',2); % 使用半径为5的圆形结构元素
% closed_image = imclose(binary_image, se);
% closed_image = bwareaopen(closed_image, 30);
% %closed_image = imgaussfilt(double(closed_image), 1);  % 高斯滤波平
% 
% % 3. 保留最大连通区域
% labeled_image = bwlabel(closed_image); % 标记连通区域
% stats = regionprops(labeled_image, 'Area');
% area_values = [stats.Area];
% [~, max_index] = max(area_values); % 找到最大连通区域的索引
% processed_image = (labeled_image == max_index); % 提取最大区域
% 
% masked_image = uint8(I);  % 复制原始图
% masked_image(~processed_image) = mean_value; % 赋值遮盖区域为 135
% 
% h = fspecial('gaussian', [3 3], 0.85);  % 创建3x3高斯滤波器
% final_image = imfilter(masked_image, h, 'replicate');
% final_image = uint8(final_image);
% % 
% % I2=double(masked_image);
% % Ivessel2=FrangiFilter2D(I2,options);
% 
% % maskoriginalimage
% % masked_image = I;  % 复制原始图像
% % masked_image(~processed_image) = 135; % 赋值遮盖区域为 135
% % final_image = uint8((255-uint8(Ivessel2*255))*0.6);
% close;
% figure;
% subplot(2, 2, 1); imshow(I,[]); title('Predicted image');
% subplot(2, 2, 2); imshow(Ivessel); title('Denoised image');
% subplot(2, 2, 4); imshow(final_image,[]); title('Processed image');
% subplot(2, 2, 3); imshow(closed_image); title('Close image');
% clear;
% % % 
% 
% end

function final_image = process_image(path, mean_value)
    I = double(imread(path));
    options = struct('FrangiScaleRange', [1 8], 'FrangiScaleRatio', 1, ...
                    'FrangiBetaOne', 1, 'FrangiBetaTwo', 8, ...
                    'verbose', true, 'BlackWhite', true);

    Ivessel = FrangiFilter2D(I, options);
    
    % 1. 图像二值化
    threshold = graythresh(Ivessel);
    binary_image = imbinarize(Ivessel, threshold);

    % 2. 形态学操作
    se = strel('disk', 2);
    closed_image = imclose(binary_image, se);
    closed_image = bwareaopen(closed_image, 10);

%     % 3. 标记连通区域
%     labeled_image = bwlabel(closed_image);
%     stats = regionprops(labeled_image, 'Area');
%     area_values = [stats.Area];
%     
%     % 找到最大连通区域
%     [~, max_index] = max(area_values);
%     max_region = (labeled_image == max_index);
%     
%     % 设置膨胀参数（可根据需要调整半径）
%     se_dilate = strel('disk', 5); % 控制邻近区域的包含范围
%     dilated_region = imdilate(max_region, se_dilate);
%     
%     % 找到所有与膨胀区域相交的连通区域
%     all_labels = 1:max(labeled_image(:));
%     retain_labels = false(size(all_labels));
%     
%     for k = all_labels
%         region_mask = (labeled_image == k);
%         if any(region_mask & dilated_region, 'all')
%             retain_labels(k) = true;
%         end
%     end
%     
%     processed_image = ismember(labeled_image, find(retain_labels));

    % 4. 应用掩膜和滤波
    masked_image = uint8(I);
    masked_image(~closed_image) = mean_value;
    
%     h = fspecial('gaussian', [3 3], 0.85);
%     final_image = imfilter(masked_image, h, 'replicate');
%     final_image = uint8(final_image);
     
    final_image = uint8(masked_image);
%     % 可视化结果
%     close;
%     figure;
%     subplot(2, 2, 1); imshow(I, []); title('Original Image');
%     subplot(2, 2, 2); imshow(Ivessel); title('Vessel Enhanced');
%     subplot(2, 2, 3); imshow(processed_image); title('Selected Regions');
%     subplot(2, 2, 4); imshow(final_image, []); title('Final Result');
%     clear;
end