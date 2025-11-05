function [masked_image, filled_image] = deVessel(I_ori) % 修改为双输出
I = double(I_ori);

gamma_value = 0.8;  % gamma < 1 会使图像变亮；这里选择 0.5 作为示例
I = imadjust(I/255, [], [], gamma_value) * 255;
    
% Frangi滤波流程保持不变
options = struct('FrangiScaleRange', [1 8], 'FrangiScaleRatio', 1, ...
    'FrangiBetaOne', 2, 'FrangiBetaTwo', 5, 'verbose',true,'BlackWhite',true);
Ivessel = FrangiFilter2D(I, options);

% 类型转换
I = uint8(I);
Ivessel = uint8(Ivessel*255);

% 二值化与形态学操作
threshold = graythresh(Ivessel);
binary_image = imbinarize(Ivessel, threshold);
se = strel('disk',3);

closed_image = imclose(binary_image, se);
closed_image = bwareaopen(closed_image, 40);

se = strel('disk',2);
closed_image = imdilate(closed_image, se);

% 生成原始遮罩图像
masked_image = uint8(I_ori);
masked_image(closed_image) = 0;

% ===== 新增局部均值填充模块 =====
filled_image = masked_image;  % 复制原始遮罩图像
filled_image = double(filled_image); % 转换为double类型以支持浮点运算

% 创建遮罩区域标记（需要填充的位置）
fill_mask = (filled_image == 0);

% 预计算扩展图像处理边界问题
pad_size = 30; % 50x50窗口半径
padded_img = padarray(filled_image, [pad_size pad_size], 'symmetric');

% 遍历所有需要填充的像素
[row, col] = find(fill_mask);
for k = 1:length(row)
    r = row(k);
    c = col(k);
    
    % 提取50x50窗口（考虑边界扩展）
    window = padded_img(r : r+2*pad_size, c : c+2*pad_size);
    
    % 计算非零均值
    non_zero_vals = window(window ~= 0);
    if ~isempty(non_zero_vals)
        filled_image(r, c) = mean(non_zero_vals);
    else
        filled_image(r, c) = 0; % 无有效像素时保持0
    end
end

filled_image = uint8(filled_image); % 转换回uint8

end
