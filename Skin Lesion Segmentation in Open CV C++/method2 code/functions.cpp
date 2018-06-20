#include <iostream>
#include "functions.h"

cv::Mat aia::project0::faceRectangles(const cv::Mat & frame) throw (aia::error)
{
	// find faces
	std::vector < cv::Rect > faces = aia::faceDetector(frame);

	// for each face found...
	cv::Mat out = frame.clone();
	for(int k=0; k<faces.size(); k++)
		cv::rectangle(out, faces[k], cv::Scalar(0, 0, 255), 2);

	return out;
}
cv::Mat aia::project0::enhanceEdges(const cv::Mat & img)
{
	cv::Mat dstImg,dx,dy;
	cv::Sobel(img, dx, CV_32F, 1,0);
	 cv::Sobel(img, dy, CV_32F, 0,1);
	 cv::magnitude(dx,dy,dstImg);
	return dstImg;

}

double aia::project0::accuracy(
		std::vector <cv::Mat> & segmented_images,		// (INPUT)  segmentation results we want to evaluate (1 or more images, treated as binary)
		std::vector <cv::Mat> & groundtruth_images,     // (INPUT)  reference/manual/groundtruth segmentation images
		std::vector <cv::Mat> & mask_images,			// (INPUT)  mask images to restrict the performance evaluation within a certain region
		int number,
		std::vector <cv::Mat> * visual_results		// (OUTPUT) (optional) false color images displaying the comparison between automated segmentation results and groundtruth
		                                                //          True positives = blue, True negatives = gray, False positives = yellow, False negatives = red
		) throw (aia::error)
	{
		// (a lot of) checks (to avoid undesired crashes of the application!)
		if(segmented_images.empty())
			throw aia::error("in accuracy(): the set of segmented images is empty");
		if(groundtruth_images.size() != segmented_images.size())
			throw aia::error(aia::strprintf("in accuracy(): the number of groundtruth images (%d) is different than the number of segmented images (%d)", groundtruth_images.size(), segmented_images.size()));
		if(mask_images.size() != segmented_images.size())
			throw aia::error(aia::strprintf("in accuracy(): the number of mask images (%d) is different than the number of segmented images (%d)", mask_images.size(), segmented_images.size()));
		for(size_t i=0; i<number ; i++)  //i<segmented_images.size()
		{
			if(segmented_images[i].depth() != CV_8U || segmented_images[i].channels() != 1)
				throw aia::error(aia::strprintf("in accuracy(): segmented image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, ucas::imdepth(segmented_images[i].depth()), segmented_images[i].channels()));
			if(!segmented_images[i].data)
				throw aia::error(aia::strprintf("in accuracy(): segmented image #%d has invalid data", i));
			if(groundtruth_images[i].depth() != CV_8U || groundtruth_images[i].channels() != 1)
				throw aia::error(aia::strprintf("in accuracy(): groundtruth image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, ucas::imdepth(groundtruth_images[i].depth()), groundtruth_images[i].channels()));
			if(!groundtruth_images[i].data)
				throw aia::error(aia::strprintf("in accuracy(): groundtruth image #%d has invalid data", i));
			if(mask_images[i].depth() != CV_8U || mask_images[i].channels() != 1)
				throw aia::error(aia::strprintf("in accuracy(): mask image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, ucas::imdepth(mask_images[i].depth()), mask_images[i].channels()));
			if(!mask_images[i].data)
				throw aia::error(aia::strprintf("in accuracy(): mask image #%d has invalid data", i));
			if(segmented_images[i].rows != groundtruth_images[i].rows || segmented_images[i].cols != groundtruth_images[i].cols)
				throw aia::error(aia::strprintf("in accuracy(): image size mismatch between %d-th segmented (%d x %d) and groundtruth (%d x %d) images", i, segmented_images[i].rows, segmented_images[i].cols, groundtruth_images[i].rows, groundtruth_images[i].cols));
			if(segmented_images[i].rows != mask_images[i].rows || segmented_images[i].cols != mask_images[i].cols)
				throw aia::error(aia::strprintf("in accuracy(): image size mismatch between %d-th segmented (%d x %d) and mask (%d x %d) images", i, segmented_images[i].rows, segmented_images[i].cols, mask_images[i].rows, mask_images[i].cols));
		}

		// clear previously computed visual results if any
		if(visual_results)
			visual_results->clear();

		// True positives (TP), True negatives (TN), and total number N of pixels are all we need
		double TP = 0, TN = 0, N = 0;
		
		// examine one image at the time
		for(size_t i=0; number ; i++)    //i<segmented_images.size()
		{
			// the caller did not ask to calculate visual results
			// accuracy calculation is easier...
			if(visual_results == 0)
			{
				for(int y=0; y<segmented_images[i].rows; y++)
				{
					aia::uint8* segData = segmented_images[i].ptr<aia::uint8>(y);
					aia::uint8* gndData = groundtruth_images[i].ptr<aia::uint8>(y);
					aia::uint8* mskData = mask_images[i].ptr<aia::uint8>(y);

					for(int x=0; x<segmented_images[i].cols; x++)
					{
						if(mskData[x])
						{
							N++;		// found a new sample within the mask

							if(segData[x] && gndData[x])
								TP++;	// found a true positive: segmentation result and groundtruth match (both are positive)
							else if(!segData[x] && !gndData[x])
								TN++;	// found a true negative: segmentation result and groundtruth match (both are negative)
						}
					}
				}
				std::cout<<i<<" Jaccard = "<< (TP + TN)/N << "\n";
			}
			else
			{
				// prepare visual result (3-channel BGR image initialized to black = (0,0,0) )
				cv::Mat visualResult = cv::Mat(segmented_images[i].size(), CV_8UC3, cv::Scalar(0,0,0));

				for(int y=0; y<segmented_images[i].rows; y++)
				{
					aia::uint8* segData = segmented_images[i].ptr<aia::uint8>(y);
					aia::uint8* gndData = groundtruth_images[i].ptr<aia::uint8>(y);
					aia::uint8* mskData = mask_images[i].ptr<aia::uint8>(y);
					aia::uint8* visData = visualResult.ptr<aia::uint8>(y);

					for(int x=0; x<segmented_images[i].cols; x++)
					{
						if(mskData[x])
						{
							N++;		// found a new sample within the mask

							if(segData[x] && gndData[x])
							{
								TP++;	// found a true positive: segmentation result and groundtruth match (both are positive)

								// mark with blue
								visData[3*x + 0 ] = 255;
								visData[3*x + 1 ] = 0;
								visData[3*x + 2 ] = 0;
							}
							else if(!segData[x] && !gndData[x])
							{
								TN++;	// found a true negative: segmentation result and groundtruth match (both are negative)

								// mark with gray
								visData[3*x + 0 ] = 128;
								visData[3*x + 1 ] = 128;
								visData[3*x + 2 ] = 128;
							}
							else if(segData[x] && !gndData[x])
							{
								// found a false positive

								// mark with yellow
								visData[3*x + 0 ] = 0;
								visData[3*x + 1 ] = 255;
								visData[3*x + 2 ] = 255;
							}
							else
							{
								// found a false negative

								// mark with red
								visData[3*x + 0 ] = 0;
								visData[3*x + 1 ] = 0;
								visData[3*x + 2 ] = 255;
							}
						}
					}
				}

				visual_results->push_back(visualResult);
			}
		}

		return (TP + TN) / N;	// according to the definition of Accuracy
	}
std::vector < cv::Mat > aia::project0::getImagesInFolder(std::string folder, std::vector<std::string> &filesNames , std::string ext, bool force_gray, bool force_bgr ) throw (aia::error)

{
	// check folders exist
	if(!ucas::isDirectory(folder))
		throw aia::error(aia::strprintf("in getImagesInFolder(): cannot open folder at \"%s\"", folder.c_str()));

	// get all files within folder
	std::vector < std::string > files;
	cv::glob(folder, files);

	// open files that contains 'ext'
	std::vector < cv::Mat > images;
	for(auto & f : files)
	{
		if(f.find(ext) == std::string::npos)
			continue;

		if(force_bgr)
			images.push_back(cv::imread(f));
		else
			images.push_back(cv::imread(f, force_gray ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_UNCHANGED));
		filesNames.push_back(f);
	}

	return images;
}

cv::Mat aia::project0::blackSuppressing(cv::Mat img){
	for(int i=0; i< img.rows; i++)
	{

		aia::uint8* imgCol = img.ptr<aia::uint8>(i);
		for (int j=0; j < img.cols; j++)
			if(imgCol[j] <= 10)
				imgCol[j] = 255 - imgCol[j];

	}
	return img;
}
double aia::project0::Jaccard( const cv::Mat & binaryImag, const cv::Mat & binaryGroundTruth)throw(aia::error){

double accuracy=0;
double TP=0, FP=0, FN =0;

double rows = binaryImag.rows;
double cols = binaryImag.cols;

if(binaryImag.rows*binaryImag.cols != binaryGroundTruth.rows*binaryGroundTruth.cols)
			throw aia::error(aia::strprintf("in accuracy(): sizes are different (%d)", binaryImag.size(), binaryGroundTruth.size()));

for(int i=0; i<rows; i++)
{
	const aia::uint8 * imgRow = binaryImag.ptr<aia::uint8>(i);
	const aia::uint8 * groundRow = binaryGroundTruth.ptr<aia::uint8>(i);
	for(int j=0; j<cols; j++)
	{
		//True positive
		if(imgRow[j] && groundRow[j])
			TP++;
		//False positive
		if(imgRow[j] && !groundRow[j])
			FP++;
		//False negative
		if(!imgRow[j] && groundRow[j])
			FN++;
	}
	
}
return TP/(TP + FP + FN);
}

/* This function takes a BGR image, reduces color diversity (reduces number of differrent colors 
by unifying neighbours in place and color to the same color */
/* ARGUMENTS :
frame: a BGR image
RETURNS:
a gray scale image
*/
cv::Mat aia::project0::cartoonify(const cv::Mat & frame) throw (aia::error)
{
	double minV, maxV;
	cv::Mat img, gray ;
	// BGRize the image
	cv::cvtColor(frame, img, cv::COLOR_GRAY2BGR);

	//do mean shift clustering
	cv::pyrMeanShiftFiltering(img, img, 30, 30, 1);  //was 30,30
	cv::cvtColor(img, gray, CV_BGR2GRAY);

	return gray;
}

cv::Mat aia::project0::inpaintHair(const cv::Mat & hairyImage)
{
			cv::Mat luminance, colorImageCopy, openedImage, hairMask;   //originalBinarized
			cv::Mat openKernel = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(30,30));
			double minVal, maxVal;

			luminance = hairyImage.clone();
			cv::minMaxLoc(luminance, &minVal, &maxVal);
			//intensities inversion
			luminance = maxVal - luminance;
//			//cv::normalize(luminance, luminance, 0, 255, cv::NORM_MINMAX);
			//make it brighter
			cv::convertScaleAbs(luminance, luminance, 2,0);

			cv::threshold(luminance, luminance, 150 , 255, CV_THRESH_OTSU | CV_THRESH_BINARY); //

			//top hat to create hair mask
			cv::morphologyEx(luminance, openedImage, cv::MORPH_OPEN, openKernel);
			hairMask = luminance - openedImage;

			cv::morphologyEx(hairMask, hairMask, cv::MORPH_CLOSE, openKernel); //was soft kernel

			// TELEA method is used to fill hair with neighbouring intensities
			cv::inpaint(hairyImage, hairMask , colorImageCopy, 37, CV_INPAINT_TELEA); //was 40
			return colorImageCopy;

}


cv::Mat aia::project0::fillBorders (const cv::Mat & grayImage){
		if(!grayImage.data)
			throw aia::error("Cannot open BGRimage");

		float scaling_factor = 0.5;
		cv::Mat orgImage  = grayImage.clone();
		cv::Mat image = grayImage.clone();
		//to use for the predicate, to fill intensities <= 135 with a brighter intensity
		int thresholdPred1 = 135;  

		// generate seeds image, which are 4 white pixels in the four corners
		cv::Mat seeds (image.rows, image.cols, CV_8U, cv::Scalar(0));
		seeds.at<unsigned char>(0,0) = 255;
		seeds.at<unsigned char>(0,image.cols-1) = 255;
		seeds.at<unsigned char>(image.rows-1,0) = 255;
		seeds.at<unsigned char>(image.rows-1,image.cols-1) = 255;

		// generate predicate 1 image
		cv::Mat img_pred_1;
		cv::threshold(image, img_pred_1, thresholdPred1, 255, CV_THRESH_BINARY_INV);

		//for  removing one-pixel black tile from some non_microscopic images
		cv::morphologyEx(img_pred_1, img_pred_1, cv::MORPH_DILATE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));

		//for cutting routes leading to image's heart!!
		cv::morphologyEx(img_pred_1, img_pred_1, cv::MORPH_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(30,30))); //was 30,30

		// region growing
		cv::Mat seeds_prev;
		do
		{
			seeds_prev = seeds.clone();
			cv::dilate(seeds, seeds, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50,50)));
			cv::Mat candidates = seeds - seeds_prev;
			seeds = seeds_prev + candidates & img_pred_1;

		}
		while( cv::countNonZero(seeds - seeds_prev) > 0);
	
		orgImage.setTo(cv::Scalar(180), seeds);// to try 255
		cv::destroyAllWindows();
		return orgImage;
}

cv::Mat aia::project0::waterShed(const cv::Mat & colorImage, const cv::Mat & binaryImage)
{
		
		cv::Mat binaryImageCopy = binaryImage.clone();
		cv::Mat colorImageCopy = colorImage.clone(), segmentedImage(binaryImage.rows, binaryImage.cols, CV_8U, cv::Scalar(0));
		if(!colorImage.data)
			throw aia::error("Cannot open image");
		

		// remove small objects (noise)
		cv::morphologyEx(binaryImageCopy, binaryImageCopy, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));
		

		// calculate distance transform
		cv::Mat dist;
		cv::distanceTransform(binaryImageCopy, dist, CV_DIST_L2, CV_DIST_MASK_3);

		// convert distance transform output (real-valued matrix) into a [0,255]-valued image
		// 'dist' values are in [min, max] --> let's find 'min' and 'max' !
		double min, max;
		cv::minMaxLoc(dist, &min, &max);
		dist = dist - min;					// shift values to the right --> values will be in [0, max-min]
		dist = dist * (255 / (max-min));	// rescale values            --> values will be in [0, 255]
		dist.convertTo(dist, CV_8U);
		

		// we want to build internal markers using the distance transform output
		// we need a 'reasonable' binarization to select only the most internal points
		cv::threshold(dist, dist, 180, 255, CV_THRESH_BINARY);
		

		// we are now ready to extract the internal markers = connected components from the previous step
		std::vector <std::vector <cv::Point> > internal_markers;
		cv::findContours(dist, internal_markers, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		
		cv::Mat markers(colorImageCopy.rows, colorImageCopy.cols, CV_32S, cv::Scalar(0));	// initialize all pixels with '0'

		// we have internal markers --> we can insert them into the 'markers' image
		for(int k=0; k<internal_markers.size(); k++)
			cv::drawContours(markers, internal_markers, k, cv::Scalar(k+1), CV_FILLED);

		std::cout<<internal_markers.size()<<"\t intern \n";
		
		// ...this is an integer-valued image with values up to billions, it's not suited for visualization!
		// if we really want to visualize it, we have to rescale it
		// by construction, 'markers' has values in [0, internal_markers.size()]
		cv::Mat markers_vis = markers.clone();
		markers_vis = markers_vis * (255.0 / (internal_markers.size()));
		markers_vis.convertTo(markers_vis, CV_8U);
		
		// we build external markers by performing a (big) dilation of the previously generated binary image
		cv::dilate(binaryImageCopy, binaryImageCopy, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(21,21)));
		

		// we are now ready to extract the external markers = connected components from the previous step
		std::vector <std::vector <cv::Point> > external_markers;
		cv::findContours(binaryImageCopy, external_markers, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		std::cout<<external_markers.size()<<"\t extern \n";

		// we have external markers --> we can insert them into the 'markers' image
		cv::drawContours(markers, external_markers, 0, cv::Scalar(internal_markers.size()+1));	// we use a label we are sure we did not used before
		
		
		// finally, we can perform the watershed
		cv::watershed(colorImageCopy, markers);
		//            /\
		//            || why on the original color image? Because OpenCV (and also MATLAB) implements the
		//               "Meyer, F. Color Image Segmentation, ICIP92, 1992" paper that can deal with colors

		markers.convertTo(markers, CV_8U, 255, 255);
		// image is now 'all white' except for the dams ('black') --> we have a binary image!

		// we can now find the contours on the inverted binary image
		std::vector < std::vector <cv::Point> > segmented_objects;
		markers = 255-markers;
		cv::findContours(markers, segmented_objects ,CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE);//CV_RETR_LIST

		// and overimpose them on the original image
		cv::drawContours(colorImageCopy, segmented_objects, -1, cv::Scalar(0,255,255), 2, CV_AA);
		cv::drawContours(segmentedImage, segmented_objects, 0, cv::Scalar(255), CV_FILLED);
		
		
		return segmentedImage;
}

int aia::project0::showHist (const cv::Mat src)
{
  /// Load image

  if( !src.data )
    { return -1; }

  /// Separate the image in 3 places ( B, G and R )
  std::vector<cv::Mat> bgr_planes;
  split( src, bgr_planes );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  cv::Mat hist;

  /// Compute the histograms:
  calcHist( &bgr_planes[0], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       cv::Scalar( 255, 0, 0), 2, 8, 0  );
      
  }

  /// Display
  //cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  aia::imshow("calcHist Dem", histImage,true);
  return 0;
}