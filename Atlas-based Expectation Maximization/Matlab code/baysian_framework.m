%% Second Part: Atlas with Tissue models, then merged
close all; clc, clear;

%histogram bins
Nbins= 4096;

% labels used for diffferent regions
labels = [0 1 2 3 ];

% training images and labels for computing the tissue model (per-region
% distribution)
imgs_path           = 'dataset\training-set\training-images\*.nii*';
labels_path         = 'dataset\training-set\training-labels\*.nii*';

% registered templates
reg_template_path   = "dataset\MNITemplateAtlas\registered-templates\";

% test images and labels for EM and DSC calculation
test_imgs_path      = "dataset\test\testing-images\";
test_labels_path    = "dataset\test\testing-labels\";

% here the instruction is commented to fasten the process by loading the
% previously-saved histograms, please feel free to check

% probs               = tissue_model(imgs_path,labels_path, [1,2,3]);

prob_struct = load('atlas_tissue_models.mat');
probs = prob_struct.probs;
% 
figure, plot(probs', 'LineWidth', 2), legend("CSF", "White matter", "Gray matter"), xlabel("rescaled intensity"), ylabel("likelihood")  

%loading registered templates
reg_template_dir = dir(reg_template_path);

% to skip "." and ".." folders
reg_template_dir = reg_template_dir(~ismember({reg_template_dir.name},{'.', '..'}));
test_labels_dir = dir(test_labels_path+'\*.nii*'); 

% ceating variables for saving the dices to an Excel file
dice_prob_atlas = zeros(length(reg_template_dir),3);
dice_liklihood  = zeros(length(reg_template_dir),3);
dice_baysian    = zeros(length(reg_template_dir),3);
dice_EM         = zeros(length(reg_template_dir),3);
dice_EM_atlas   = zeros(length(reg_template_dir),3); 
img_name        = cell(length(reg_template_dir),1);

for i=1:length(reg_template_dir)
    struct = dir(reg_template_dir(i).folder + "\" + reg_template_dir(i).name + "\*.1.nii*");
    
    % %load the corresponding label
    idx = test_labels_dir(contains({test_labels_dir.name}, {reg_template_dir(i).name})).name;  %'*'+test_imgs_dir(i).name}).name+'*'
    disp("loading label image " + idx)
    test_labels = double(niftiread(test_labels_dir(i).folder + "\" + idx));
    
    % loading the test image and the registered template-labels
    disp("loading test image " + reg_template_dir(i).name)
    test_img    = niftiread(test_imgs_path + reg_template_dir(i).name + ".nii.gz" );
    
    disp("loading " + struct.folder + '\' + 'label1')
    label1      =   double(niftiread(struct.folder + "\" + "label1\result.nii.gz"));
    
    disp("loading " + struct.folder + '\' + 'label2')
    label2      =   double(niftiread(struct.folder + "\" + "label2\result.nii.gz"));
    
    disp("loading " + struct.folder + '\' + 'label3')
    label3      =   double(niftiread(struct.folder + "\" + "label3\result.nii.gz"));
    
    %normalizing the test image to fit in the histogram
    rescaled_img = uint16(rescale_img(test_img, Nbins-1));

    %finding p(y|x) for each pixel, put it in the corresponding position in
    %conditional probability matrix (1 Cerobspinal cord, 2 GRAY matter, 3 WHITE
    %matter)
    [row, col, depth] = size(rescaled_img);

    % killing labels outside the predefined range (labels defined above)
    if ~isempty(setdiff(unique(test_labels), labels))
        disp("violating labels" + struct.folder(find(struct.folder== '\',1, 'last')+1:end))
        non_intersection = setdiff(unique(test_labels), labels);
        for k=1:length(non_intersection)
            test_labels(test_labels==non_intersection(k))=0;
        end
    end
    
    % defining the region of interest as tissue pixels only (zero is
    % reserved for background as convention)
    ROI = test_labels~=0;
    
    % getting the tissue probabilities at image intensities (+1 so no zero
    % values 1 -> Nbins)
    cond_prob1 = reshape(probs(1,rescaled_img+1), row, col, depth);
    cond_prob2 = reshape(probs(2,rescaled_img+1), row, col, depth);
    cond_prob3 = reshape(probs(3,rescaled_img+1), row, col, depth);

    %multiply the conditional probbility with the corresponding prior
    %probbility in probabilistic atlas

    pst_prob1 = label1 .* cond_prob1;
    pst_prob2 = label2 .* cond_prob2;
    pst_prob3 = label3 .* cond_prob3;
    
    
    %argmax for the posterior porbabilities (4th dimension)
    
    % baysian probabilities
    [~,pred_labels] = max(cat(4, pst_prob1, pst_prob2, pst_prob3),[],4);
    pred_labels(~ROI)=0;

    % tissue model probs
    [~,pred_labels_liklihood] = max(cat(4, cond_prob1, cond_prob2, cond_prob3),[],4);  %label1, label2, label3
    pred_labels_liklihood(~ROI)=0;
    
    % probabilistic atlas probs
    [~,pred_labels_atlas] = max(cat(4, label1, label2, label3),[],4);  %label1, label2, label3
    pred_labels_atlas(~ROI)=0;
    
    %EM probabilities (initialized with baysian labels)
    [EM_csf_prob, EM_wm_prob, EM_gm_prob] = EM_Probability(test_img, pred_labels);
    [~, pred_EM_probs] = max(cat(4, EM_csf_prob, EM_wm_prob, EM_gm_prob), [], 4);
    pred_EM_probs(~ROI)= 0;
    
    
    %Atlas as prior and EM as lilkiihood
    em_atlas_csf    = EM_csf_prob   .*  pst_prob1;
    em_atlas_wm     = EM_wm_prob    .*  pst_prob2;
    em_atlas_gm     = EM_gm_prob    .*  pst_prob3;
    
    [~, pred_EM_atlas] = max(cat(4, em_atlas_csf, em_atlas_wm, em_atlas_gm), [], 4);
    pred_EM_atlas(~ROI)= 0;
        
    
    %% displaying dices
    disp("displaying dices for image " + struct.folder(find(struct.folder== '\',1, 'last')+1:end))
    
    %finding the DSC
    disp("probabilistic Atlas dices (CSF, WM, GM)")
    dice_prob_atlas(i,:) = Dice_metric(pred_labels_atlas, test_labels)';
    dice_prob_atlas(i,:)

    disp("liklihood dices (CSF, WM, GM) ")
    dice_liklihood(i,:) = Dice_metric(pred_labels_liklihood, test_labels)';
    dice_liklihood(i,:)

    disp("Baysian framework dices (CSF, WM, GM) ")
    dice_baysian(i,:) = Dice_metric(pred_labels, test_labels)';
    dice_baysian(i,:)
    
    img_name{i,1} = string(reg_template_dir(i).name);
    
    disp("displaying dices for EM result for image " + img_name{i,1})
    dice_EM(i,:) = Dice_metric(pred_EM_probs, test_labels)';
    dice_EM(i,:)
    
    disp("displaying dices for EM_Atlas result for image " + img_name{i,1})
    dice_EM_atlas(i,:) = Dice_metric(pred_EM_atlas, test_labels)';
    dice_EM_atlas(i,:)
    
end

%% saving to an Excel file and plotting Box plots
tab = table(img_name, dice_prob_atlas, dice_liklihood, dice_baysian, dice_EM, dice_EM_atlas );
writetable(tab, 'results.xlsx')

figure,
boxplot([dice_prob_atlas(:,1), dice_liklihood(:,1), dice_baysian(:,1), dice_EM(:,1), dice_EM_atlas(:,1)],...
    'Labels', {'Prboabilstic Atlas','Tissue model','Baysian Framework','EM', 'Baysian EM'})...
    , title("CSF segmentation Using Small Atlas")...
    ,ylabel("DSC"), grid, set(gca,'XTick',[1,2,3,4,5]), set(gca,'YTick', 0:0.05:1)

figure,
boxplot([dice_prob_atlas(:,2), dice_liklihood(:,2), dice_baysian(:,2), dice_EM(:,2), dice_EM_atlas(:,2)],...
    'Labels', {'Prboabilstic Atlas','Tissue model','Baysian Framework','EM', 'Baysian EM'})...
    , title("White-matter segmentation Using Small Atlas")...
    ,ylabel("DSC"), grid, set(gca,'XTick',[1,2,3,4,5]), set(gca,'YTick', 0:0.05:1)

figure,
boxplot([dice_prob_atlas(:,3), dice_liklihood(:,3), dice_baysian(:,3), dice_EM(:,3), dice_EM_atlas(:,3)],...
    'Labels', {'Prboabilstic Atlas','Tissue model','Baysian Framework','EM', 'Baysian EM'})...
    , title("Grey-matter segmentation Using Small Atlas")...
    ,ylabel("DSC"), grid, set(gca,'XTick',[1,2,3,4,5]), set(gca,'YTick', 0:0.05:1)
