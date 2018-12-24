function Dice_vec = Dice_metric(label_mat1, label_mat2)
%Dice(label_mat1, label_mat2)
%This function takes two same-size label matrices and calculates the dice
%value for each region.
%
%%Arguments:
%label_mat1: first label matrix 
%label_mat2: second label matrix

% checking sizes
if size(label_mat1)~= size(label_mat2)
    error("mismatched sizes of label matrices");
end

% detecting regions
labels1 = unique(label_mat1);   
labels2 = unique(label_mat2);

% ensuring that all regions in both matrices are detected 
if length(labels1) >= length(labels2)
    labels = labels1;
else
    labels = labels2;
end

num_regions = length(labels)-1; % 0 is allocated for the background
Dice_vec = zeros(num_regions,1);
intersect_count = zeros(num_regions,1); % intersection
region_pix_count1 = zeros(num_regions,1); % number of pixels belonging to mat1
region_pix_count2 = zeros(num_regions,1); % number of pixels belonging to mat2



[rows, cols, depth] = size(label_mat1);
for k=1:depth
    for i=1:rows
        for j=1:cols
            %region assignment
            if label_mat1(i,j,k) ~=0
                region_pix_count1(label_mat1(i,j,k)) = region_pix_count1(label_mat1(i,j,k)) +1;
            end
            if label_mat2(i,j,k) ~=0
                region_pix_count2(label_mat2(i,j,k)) = region_pix_count2(label_mat2(i,j,k)) +1;
                %intersection detection
                if label_mat1(i,j,k) == label_mat2(i,j,k)
                    intersect_count(label_mat1(i,j,k)) = intersect_count(label_mat1(i,j,k)) +1; 
                end
            end
        end
    end
end

for i=1:num_regions
   Dice_vec(i) = 2*intersect_count(i)/(region_pix_count1(i) + region_pix_count2(i)); 
end

%checked on 17-Oct-18 at 16:27