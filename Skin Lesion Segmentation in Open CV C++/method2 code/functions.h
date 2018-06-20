#ifndef _project_0_h
#define _project_0_h

#include "aiaConfig.h"
#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\photo\photo.hpp>
#include "ucasConfig.h"

// open namespace "aia"
namespace aia
{
	// define a new namespace "project0"  within "aia"
	namespace project0
	{
		// this is just an example: find all faces in the given image using HaarCascade Face Detection
		cv::Mat faceRectangles(const cv::Mat & frame) throw (aia::error);
		cv::Mat enhanceEdges(const cv::Mat & img);
		cv::Mat findShape(const cv::Mat & img);
		std::vector < cv::Mat > getImagesInFolder(std::string folder, std::vector<std::string> &filesNames, std::string ext = ".tif",  bool force_gray = false, bool fore_bgr =false);
	
		double accuracy(
		std::vector <cv::Mat> & segmented_images,		// (INPUT)  segmentation results we want to evaluate (1 or more images, treated as binary)
		std::vector <cv::Mat> & groundtruth_images,     // (INPUT)  reference/manual/groundtruth segmentation images
		std::vector <cv::Mat> & mask_images,			// (INPUT)  mask images to restrict the performance evaluation within a certain region
		int number,
		std::vector <cv::Mat> * visual_results = 0		// (OUTPUT) (optional) false color images displaying the comparison between automated segmentation results and groundtruth
		                                                //          True positives = blue, True negatives = gray, False positives = yellow, False negatives = red
		) throw (aia::error);
		cv::Mat blackSuppressing( cv::Mat img);
		double Jaccard ( const cv::Mat& binaryImag, const cv::Mat  &binaryGroundTruth) throw (aia::error);
		cv::Mat cartoonify(const cv::Mat & frame) throw (aia::error);

		/*!This function takes an image and does some morphological operations and inpaints hair locations */
		cv::Mat inpaintHair(const cv::Mat & hairyImage);

		/*!This function fills black borders (when available) with intensities relative to image content*/
		/*ARGUMENTS:
		image: a BGR image*/
		cv::Mat fillBorders (const cv::Mat & image);

		/* This function tries to finely segment the lesion preliminarily segmented*/
		/* ARGUMENTS:
		colorImage: BGR image
		binaryLesionImage: is the initial position for the lesion, lesion: white, background : black
		*/
		cv::Mat waterShed (const cv::Mat & colorImage, const cv::Mat & binaryLesionImage);

		int showHist (cv::Mat src);
	}

}

#endif // _project_0_h

