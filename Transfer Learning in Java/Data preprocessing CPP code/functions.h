#ifndef _project_0_h
#define _project_0_h

#include "aiaConfig.h"
#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>

// open namespace "aia"
namespace aia
{
	// define a new namespace "project0"  within "aia"
	namespace project0
	{

		/* MovingWindow: This function applies the mechanism of taking multiple crops with some criteria from the image
		the window size is 454*454 and the criteria are:
		- complete window fit inside the mask (white region)
		- negative windows are defined as having zero ground truth pixels.
		- positive windows are defined as have center 15*15 block all white ground truth.
		- negative stride is generally should be larger than positive slide due to tha lack of adequate positive windows.
		ARGUMENTS:
		imagesPath: the path to gray images path
		masksPath: the path to masks images
		groundTruthPath: the path to ground truth images
		negStride: negative stride used as the step of the moving window on images with no mass
		posStride: positive stride used as the step of the moving window on images with mass(es)
		trainPosImgFolderPath: the path into which training positive windows will be saved
		trainNegImgFolderPath: the path into which training negative windows will be saved

		testPosImgFolderPath: the path into which testing positive windows will be saved
		testNegImgFolderPath: the path into which testing negative windows will be saved

		trainTestRatio: the ratio between training set size and test set size.
		*/

		void movingWindow (std::string imagesPath, std::string masksPath, std::string groundTruthPath, int negstride , int posStride,
			std::string testPosImgFolderPath, std::string testNegImgFolderPath, std::string trainPosImgFolderPath, std::string trainNegImgFolderPath, float trainTestRatio);

		/* THIS FUNCTION, splitNegPos, TAKES TWO VECTORS OF FILES NAMES,i.e. GROUND TRUTH AND IMAGES NAMES, RETURNS A VECTOR OF TWO ELEMENTS
		(as the function name suggests !!!)THE FIRST ELEMENT IS THE VECTOR OF NEGATIVE (NO GROUND TRUTH) FILES NAMES,
		THE SECOND IS A VECTOR OF POSITIVE FILES NAMES( HAVE GROUND TRUTH)*/

		std::vector<std::vector<std::string>> splitNegPos(const std::vector<std::string> &images, const	std::vector<std::string> & groundTruth);

		/*This function takes a database folder and a destination folder, it takes a parameterized maximum number
				of samples from images with matched names (till the end of the pattern)
				and save them into the destination folder.
				ARGUMENTS:
				srcFolderPath: the source folder absolute path
				dstFolderPath: destination folder absolute path
				pattern: used for the mechanism of detecting similar images that share the same name till the end of the Pattern
				samplesNum: the maximum allowed number of samples beloning to the same image*/

		void reduceDataBase(std::string srcfolderPath, std::string dstFolderPath, std::string pattern, int samplesNum);
	}
}

#endif // _project_0_h
