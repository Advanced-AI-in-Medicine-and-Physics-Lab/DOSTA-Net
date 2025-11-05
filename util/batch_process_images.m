function batch_process_images(input_folder,output_root)
    % 批量处理图像并按原路径结构保存到 processed 文件夹
    %
    % 输入：
    %   input_folder - 原始图片的根目录（包含两级子目录）
    %
    % 输出：
    %   处理后的图片保存在 processed 文件夹中，保持原有目录结构

    % 获取所有二级子目录
    subdirs = dir(input_folder);
    subdirs = subdirs([subdirs.isdir] & ~ismember({subdirs.name}, {'.', '..'}));

    % 创建 processed 目录
    % output_root = fullfile(input_folder, 'processed');
    if ~exist(output_root, 'dir')
        mkdir(output_root);
    end

    % 遍历每个二级子目录
    for i = 1:numel(subdirs)
        subdir_path = fullfile(input_folder, subdirs(i).name);
        output_subdir = fullfile(output_root, subdirs(i).name);

        % 创建输出目录
        if ~exist(output_subdir, 'dir')
            mkdir(output_subdir);
        end

        % 获取当前子目录中的所有图片
        image_files = dir(fullfile(subdir_path, '*.png')); % 假设图片格式为PNG
        I=uint8(imread(fullfile(subdir_path, image_files(1).name)));
        mean_value = mean(I(:));
        for j = 1:numel(image_files)
            input_image_path = fullfile(subdir_path, image_files(j).name);
            output_image_path = fullfile(output_subdir, image_files(j).name);
            
            % 处理图片
            processed_image = process_image(input_image_path,mean_value);
            processed_image_uint8 = uint8(processed_image);
            % 保存处理后的图像
            imwrite(processed_image_uint8, output_image_path);

            fprintf('Processed and saved: %s\n', output_image_path);
        end
    end

    disp('所有图像已处理完成。');
end