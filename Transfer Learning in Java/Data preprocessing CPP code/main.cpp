// include aia and ucas utilities
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


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

