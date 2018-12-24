function [regions_dists] = tissue_model(imgs_path, labels_path, labels)
%[regions_dists] = tissue_model(imgs_dir_struct, labels_dir_struct)
%   computes the probability density function of each region using all the images included in the directrory.
%parameters:
%   imgs_path: the path where the images lie
%   labels_path: the path where the labels lie
%   labels: a vector of N values, where N is the number of regions.
%   region i has the label regions_labels(i) and the ith row in regions_dists will have the probabilites of region i. 
%   

%loading the directories
imgs_dir_struct = dir(imgs_path);
labels_dir_struct = dir(labels_path);

%initialization
len = length(imgs_dir_struct);
regions_num = length(labels);
counters = zeros(regions_num,1);

% number of bins is supposed to be representable by no more than 16 bits
Nbins = 4096   
regions_dists= zeros(regions_num, Nbins);

%loop over all the images in the directory
for i=1:len

    % read the image and the corresponding label/ assuming they are sorted
    % accordingly
    img = (niftiread(cat(2,imgs_dir_struct(i).folder, '\', imgs_dir_struct(i).name))); 
    
    regs_labels = double(niftiread( cat(2,labels_dir_struct(i).folder, '\', labels_dir_struct(i).name)));

    disp("analysing the distribution of image "+  imgs_dir_struct(i).name)
    disp("maximum value " + num2str(max(img,[], 'all')))
    
    %normalization 0 -> Nbins-1
    img = uint16(round(rescale_img(img, Nbins-1)));
    %unique(regs_labels)
    
    %Accumulate the histograms and counters for the three regions
    for l=1:regions_num
        reg_label = regs_labels(regs_labels==l); 
        counters(l) = counters(l) + length(reg_label);
        [counts,~] = histcounts(img(regs_labels ==l), Nbins, 'BinLimits', [-0.5, Nbins-.5]);
        regions_dists(labels(l),:) = regions_dists(l,:) + counts; 
    end
    
end

% normalize each histogram by the corresponding counter
for i=1:regions_num
   regions_dists(i,:) =  regions_dists(i,:)/counters(i);
   
end

% moving-window filter to fill holes and reduce perturbations
regions_dists = movmean(regions_dists,45,2);

% intensity-wise normalization, division by sum of histograms at each
% intensity for available (non-zero) histogram values
for j=1:length(counts)
    if (sum(regions_dists(:,j) <= eps(1)))<1
        regions_dists(:, j) =  regions_dists(:,j)/sum(regions_dists(:,j));
    end

end
