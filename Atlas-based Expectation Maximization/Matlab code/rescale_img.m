function rescaled_img = rescale_img(grayImg, max_val)
%rescaled_img = rescale_img(grayImg)
%This function takes a gray image of any dimensions and returns the rescaled version using 
%the maximum and minimum values without changing the data type.
%the returned image will have a minimum of zero and a maximum of max_val

grayImg = double(grayImg);
maximum = max(grayImg,[], 'all');
% info = whos('grayImg');
% maximum = 2^(info.bytes*8/numel(grayImg));
minimum = min(grayImg, [], 'all');

if(maximum==0)
    disp('empty inputs')
    rescaled_img = grayImg;

else
    rescaled_img = max_val*((grayImg - minimum)/(maximum - minimum));
    
end

end