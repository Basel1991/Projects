function [msd_avg, msd_std] = eval_TRE(fixed_landmarks, mov_landmarks, voxel_dimensions)%*
%ssd = ssd3d(fixed_landmarks, moving_landmarks)
%   calculates the 3D sum of squared differences (Eulidean distance)
%   between two landmarks.
% Where there exist N landmarks each with M coordinates 
%Parameters:
%   fixed_landmarks: N * M double matrix
%       coordinates (in pixels) for landmarks in the fixed image
%   moving_landmarks: N * M double matrix
%       coordinates (in pixels) for landmarks in the moving image
%   voxel_dimensions: a M X 1 vector
%       the dimensions in mm for one voxel

%Returns:
%   ssd: the sum of squared differences of distances between corresponding
%   landamrks.
%

%initialization
num_features = length(fixed_landmarks);

%computing the distance in mm between correspondnig features and matches
%using the voxel spacing provided
difference = fixed_landmarks - mov_landmarks;
pixel2mm = repmat([voxel_dimensions(1), voxel_dimensions(2), voxel_dimensions(3)], num_features,1);
ssd = sqrt(sum((difference.* pixel2mm).^2,2));

%average and standard deviation for the distances
msd_avg = mean(ssd);
msd_std = std(ssd);
       
end

