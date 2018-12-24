%%This script do the following:
% reads nifti images and labels
%average nifti images with equal weights
%average different labels separately with equal weights for all pixels
%saves the results in a folder "results"
clc, close, clear;
images_folder   = "dataset\SmallAtlas\registered-images\*\*1.nii*";
labels_folder   = "dataset\SmallAtlas\registered-labels\*\*.nii*";
output_path            = "dataset\SmallAtlas\averaged";

% metricsCell = common_style(imagesFolder);
imgs_dir     = dir(images_folder);
labels_dir   = dir(labels_folder);

len             = length(imgs_dir);

img1_struct            = load_untouch_nii(char(imgs_dir(1).folder + "\" + imgs_dir(1).name));
label1_struct          = load_untouch_nii(char(labels_dir(1).folder + "\" + labels_dir(1).name));   

% this image is used only to read dimensions
img1                   = img1_struct.img;
[row, col, depth]= size(img1);

%initialization
data_base       = zeros(row, col, depth);
all_label_1     = zeros(row, col, depth);
all_label_2     = zeros(row, col, depth);
all_label_3     = zeros(row, col, depth);


for i=1:length(imgs_dir)
   
   % % loading using load_untouch_nii
   new_img_struct   = load_untouch_nii(char(imgs_dir(i).folder + "\" + imgs_dir(i).name));
   
   % %modifying to double
   new_img_struct.hdr.dime.bitpix=64;
   new_img_struct.hdr.dime.datatype=64;
   new_img          = double(new_img_struct.img);                                                     
   
   labels_struct    = load_untouch_nii(char(labels_dir(i).folder + "\" + labels_dir(i).name));
   labels_struct.hdr.dime.bitpix=64;
   labels_struct.hdr.dime.datatype=64;
   labels           = double(labels_struct.img);                                                      
   
   data_base        = data_base + new_img./len;
   
   temp             = zeros(row, col, depth);
   temp(labels==1)  = 1/len;
   all_label_1      = temp + all_label_1;
   
   temp             = zeros(row, col, depth);
   temp(labels==2)  = 1/len;
   all_label_2      = temp + all_label_2;
   
   temp             = zeros(row, col, depth);
   temp(labels==3)  = 1/len;
   all_label_3      = temp + all_label_3;
end

%% Saving using save_untouch

new_img_struct.img = data_base;
save_untouch_nii(new_img_struct, char(output_path + "\average_img.nii"));  %no untouch was with make and database_struct

% new_labels_struct       =   make_nii(all_label_1); 
labels_struct.img = all_label_1;
save_untouch_nii(labels_struct,  char(output_path + "\average_label_1.nii"));

% new_labels_struct.img   =   make_nii(all_label_2); 
labels_struct.img = all_label_2;
save_untouch_nii(labels_struct,  char(output_path + "\average_label_2.nii"));

% new_labels_struct.img   =   make_nii(all_label_3); 
labels_struct.img = all_label_3;
save_untouch_nii(labels_struct,  char(output_path + "\average_label_3.nii"));

%% showing the tissues models 
probs = tissue_model(images_folder, labels_folder, [1,2,3]);
figure, plot(probs', 'LineWidth', 2), legend("CSF", "White matter", "Gray matter"), xlabel("rescaled intensity"), ylabel("likelihood")
