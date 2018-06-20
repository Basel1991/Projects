// include aia and ucas utilities
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace
{
	// since we work with a GUI, one possible solution is to store parameters 
	// (including images) in global variables
	cv::Mat img;							// original image
	int sharpening_factor_x100; 			// 'X100' means it is multiplied by 100: 
											// unfortunately the OpenCV GUI does not support real-value 
											// trackbars, so we have to deal with integers


	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	//       see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void sharpeningCallback(int pos, void* userdata) 
	{
		// get the actual factor
		float k = 0.1 * sharpening_factor_x100;

		// Laplace filter (2nd order derivatives), invariant to 90° rotations, is
		//
		// 0   1   0
		// 1  -4   1
		// 0   1   0
		//
		// A better, 45° rotation invariant version is given by
		//
		// 1   1   1
		// 1  -8   1
		// 1   1   1
		//
		// In order to sharpen the image f using the Laplacian filter L, we need to calculate
		// fsharp = f - k·f*L,   where '*' denotes convolution, and 'k' is a constant
		// then, applying the associative property of convolution, we can write
		// fsharp = f*LK, where 'LK' is a kernel defined as
		// 
		// 0   0   0        1   1   1      -k  -k  -k
		// 0   1   0  -k·   1  -8   1   =  -k 1+8k -k
		// 0   0   0        1   1   1      -k  -k  -k
		cv::Mat LK = (cv::Mat_<float>(3, 3) <<
			-k,  -k,    -k,
			-k,  1+8*k, -k,
			-k,  -k,    -k);

		// 'filter2D' is the convolution operation
		// in this case, it is the convolution of 'f' with the kernel 'LK', 
		// and the result is stored into 'sharpened' of type 'CV_8U' ( = unsigned char = 8-bit grayscale image)
		cv::Mat sharpened;
		cv::filter2D(img, sharpened, CV_8U, LK);

		// In this case it's 100% ok to store the result into a 'CV_8U' matrix, since this operation (sharpening)
		// does not shift / stretch the histogram of the image.
		// More precisely, it may produce out-of-visualization-range values at some pixels (see the plot in your lecture's slides),
		// especially the black and white pixels close to edges, but here we decided to let them saturate by automatically converting to CV_8U.

		// In general, when the convolution filter elements do not sum to one, and/or it has negative values,
		// it can yield a result that cannot be visualized directly. In these cases, a min-max normalization step is required
		// using, for instance, cv::normalize with norm_type = cv::NORM_MINMAX

		// show result
		aia::imshow("sharpening", sharpened,true, 0.5);
	}

}


int main() 
{
	try
	{	std::string imagesFolderPath	= "G:/Google Drive/Classroom/AIA/projects/AIA-Mass-Segmentation/dataset/images";
		std::string masksFolderPath		= "G:/Google Drive/Classroom/AIA/projects/AIA-Mass-Segmentation/dataset/masks";
		std::string groundTruthFolderPath = "G:/Google Drive/Classroom/AIA/projects/AIA-Mass-Segmentation/dataset/groundtruth";

		std::string trainPosWindowsFolderPath = "G:/Windows/original/train/posWindows";
		std::string trainNegWindowsFolderPath = "G:/Windows/original/train/negWindows";
		std::string testPosWindowsFolderPath = "G:/Windows/original/test/posWindows";
		std::string testNegWindowsFolderPath = "G:/Windows/original/test/negWindows";

		/*aia::project0::movingWindow(imagesFolderPath, masksFolderPath, groundTruthFolderPath, 450,25,testPosWindowsFolderPath, testNegWindowsFolderPath,
			trainPosWindowsFolderPath, trainNegWindowsFolderPath, 0.7);*/

		//aia::project0::reduceDataBase(trainNegWindowsFolderPath , "G:/Windows/large/train/negWindows", "_", 95); 
		//aia::project0::reduceDataBase(trainPosWindowsFolderPath , "G:/Windows/large/train/posWindows", "_", 300); 

		//aia::project0::reduceDataBase(testNegWindowsFolderPath , "G:/Windows/large/test/negWindows", "_", 40); 
		//aia::project0::reduceDataBase(testPosWindowsFolderPath , "G:/Windows/large/test/posWindows", "_", 320); 

		
		cv::Mat temp = cv::imread("G:/Google Drive/Classroom/AIA/projects/AIA-Mass-Segmentation/dataset/images/24065530_d8205a09c8173f44_MG_L_ML_ANON.tif");
		cv::imwrite("G:/Google Drive/Classroom/AIA/projects/AIA-Mass-Segmentation/dataset/images/mammogram1.png", temp);
		return 1;
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}
/*//freopen("G:/Windows/output.txt","w",stdout);
		std::string imagesFolderPath	= "G:/Google Drive/Classroom/AIA/projects/AIA-Mass-Segmentation/dataset/images";
		std::string masksFolderPath		= "G:/Google Drive/Classroom/AIA/projects/AIA-Mass-Segmentation/dataset/masks";
		std::string groundTruthFolderPath = "G:/Google Drive/Classroom/AIA/projects/AIA-Mass-Segmentation/dataset/groundtruth";

		std::string posWindowsFolderPath = "G:/Windows/PositiveWindows";
		std::string negWindowsFolderPath = "G:/Windows/NegativeWindows";
		//aia::project0::movingWindow(imagesFolderPath, masksFolderPath, groundTruthFolderPath, 450,90, posWindowsFolderPath ,
			//negWindowsFolderPath);
		aia::project0::reduceDataBase(negWindowsFolderPath , "G:/Windows/balanced/negativeWindows", "_"); 
				return 1;*/

