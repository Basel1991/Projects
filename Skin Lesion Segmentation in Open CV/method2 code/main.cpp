// Editor: Basel Alyafi
// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <exception>
// include my project functions
#include "functions.h"
#include <iostream>

int main() 
{
	//For this algorithm to work fine and save images, there should be at least one folder called "lesions" inside the place 
	//of images you are working on to save the segmented images inside
	//define paths
	std::string folderPath = "G:/Google Drive/Classroom/AIA/projects/AIA-Skin-Lesion-Segmentation/dataset/images";
	std::string groundTruthPath = "G:/Google Drive/Classroom/AIA/projects/AIA-Skin-Lesion-Segmentation/dataset/groundtruths";
	
	std::string inpaintedPath = "G:/Google Drive/Classroom/AIA/projects/AIA-Skin-Lesion-Segmentation/dataset/images/inpainted";

	std::ofstream outStream; 
	std::string inpaintedFileName, groundTruthName, cartoonifiedFileName, lesionFileName, pureImageName;
	std::vector<std::string> filesNames, groundTruthNames, lesionsNames, inpaintedNames;

	int backSlashIndex;
	double processedJac=0, dummyJac=0;
		
	cv::Mat luminance, groundTruthImage, originalBinarized, inpaintedImage, filledImage;
	try
	{				
		outStream.open(folderPath + "/Jaccards.csv",std::ios::app);
		
		std::vector <cv::Mat> imagesVector = aia::project0::getImagesInFolder(folderPath, filesNames, ".jpg" );
		std::vector <cv::Mat> grayImagesVector = aia::project0::getImagesInFolder(folderPath,filesNames, ".jpg", true);
		std::vector <cv::Mat> groundTruthVector = aia::project0::getImagesInFolder(groundTruthPath,groundTruthNames, ".png");

		cv::Mat softCloseKernel = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(30,30));
		cv::Mat harsherCloseKernel = cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(30,30)), lesion; //was 40,40

		for(int i =0; i<imagesVector.size()  ; i++)  //&& i<1 
		{
			
			/////////////////////////////////////////////////////////////// filling corners and inpainting hair
			
			filledImage = aia::project0::fillBorders(grayImagesVector[i]);		
			if(i<10)
				cv::imwrite(folderPath+ "/filled/filled_" + std::to_string(i) + ".png", filledImage);

			inpaintedImage = aia::project0::inpaintHair(filledImage);
			//////////////////////////////////////////////////////////////

			backSlashIndex = filesNames[i].find("\\");
			pureImageName = filesNames[i].substr(backSlashIndex+1);
			inpaintedFileName = folderPath + "/inpainted/" + pureImageName;
			cartoonifiedFileName = folderPath + "/cartoonified/" + pureImageName;

			lesionFileName = folderPath + "/lesions/" + pureImageName;
			

			groundTruthName = filesNames[i].substr(backSlashIndex+1, filesNames[i].find_last_of('.') - backSlashIndex-1 )+ "_segmentation.png";
			cv::imwrite(inpaintedFileName, inpaintedImage);
			groundTruthImage = cv::imread(groundTruthPath+"/"+ groundTruthName, CV_LOAD_IMAGE_UNCHANGED);

			//------------------------------------------------------ mean shift
			luminance = aia::project0::cartoonify(inpaintedImage);
			cv::imwrite(cartoonifiedFileName, luminance);
			
			//do a dummy processing on original images for the sake of comparison
			cv::cvtColor( imagesVector[i],  originalBinarized, CV_BGR2GRAY);
			cv::threshold(originalBinarized, originalBinarized, 21, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
			originalBinarized = 255 - originalBinarized ;
			

			cv::threshold(luminance, luminance, 80, 255, CV_THRESH_OTSU|CV_THRESH_BINARY); //
			luminance = 255-luminance;  //inversion from binary to binary inverse
			//cv::morphologyEx(luminance, luminance, cv::MORPH_CLOSE, softCloseKernel);

			
			
			//post processing//

			//fill black holes
			cv::morphologyEx(luminance, luminance, cv::MORPH_CLOSE, harsherCloseKernel);//was softer

			//remove or reduce white exteriors
			cv::morphologyEx(luminance, luminance, cv::MORPH_OPEN, harsherCloseKernel);//was softer

			cv::imwrite(lesionFileName, luminance);

			//calculate the Jaccard index
			processedJac = aia::project0::Jaccard(luminance, groundTruthImage);
			dummyJac = aia::project0::Jaccard(originalBinarized, groundTruthImage);

			std::cout<<i<<"\t"<< processedJac <<"\n" << i << "\t" << dummyJac <<"\n";
			outStream<<pureImageName +"," + std::to_string(processedJac) +"," + std::to_string(dummyJac)<<std::endl;
			//cv::destroyAllWindows();			

		}
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
	catch(std::exception except)
	{
		std::cout<<"EXCEPTION thrown by unknown source :\n\t|=> " << except.what() <<"\n";

	}
} 
