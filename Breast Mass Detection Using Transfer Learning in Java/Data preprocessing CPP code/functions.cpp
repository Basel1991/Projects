#include <iostream>
#include "functions.h"
#include <math.h>

void aia::project0::movingWindow (std::string imagesPath, std::string masksPath, std::string groundTruthPath, int negStride,
			int posStride, std::string testPosImgFolderPath, std::string testNegImgFolderPath, std::string trainPosImgFolderPath, std::string trainNegImgFolderPath, float trainTestRatio)
{

	std::vector<cv::String> grayImagesNames ;
	std::vector<std::vector<cv::String>> negPosImagesNames;
	std::vector<cv::String> groundTruthNames;
	cv::glob(imagesPath, grayImagesNames, false); //load all images names in the folder
	cv::glob(groundTruthPath, groundTruthNames, false); //load all images names in the folder

	negPosImagesNames = splitNegPos(grayImagesNames, groundTruthNames);
	std::vector<cv::String> trainPosImages(ceil(trainTestRatio * negPosImagesNames[1].size()));
	std::vector<cv::String> testPosImages(negPosImagesNames[1].size() - trainPosImages.size());
	std::vector<cv::String> trainNegImages(ceil(trainTestRatio * negPosImagesNames[0].size()));
	std::vector<cv::String> testNegImages(negPosImagesNames[0].size() - trainNegImages.size());

	std::copy(negPosImagesNames[0].begin(), ceil(trainTestRatio * negPosImagesNames[0].size())+ negPosImagesNames[0].begin(), trainNegImages.begin());
	std::copy(negPosImagesNames[1].begin(), ceil(trainTestRatio * negPosImagesNames[1].size())+ negPosImagesNames[1].begin(), trainPosImages.begin());

	std::copy(negPosImagesNames[0].begin() + ceil(trainTestRatio * negPosImagesNames[0].size()), negPosImagesNames[0].end(), testNegImages.begin());
	std::copy(negPosImagesNames[1].begin() + ceil(trainTestRatio * negPosImagesNames[1].size()), negPosImagesNames[1].end(), testPosImages.begin());

	cv::Mat image, maskImg, groundTruthImg, window, windowCenter;
	cv::Mat GroundTruthRect;
	std::string imageName, maskAbsPath, extension, writeFileName, posImgFolderPath, negImgFolderPath;
	int dotPosition, windowSize = 454 * 454, lastBackSlash, negImageCount=0;
	int posCounter=0, negCounter=0, stride, stridesRatio = negStride/posStride;
	for(int type =0; type<2; type++)
	{
		negPosImagesNames.clear();
		if(type == 0) // train dataset
		{
			negPosImagesNames.push_back(trainNegImages);
			negPosImagesNames.push_back(trainPosImages);
			posImgFolderPath = trainPosImgFolderPath;
			negImgFolderPath = trainNegImgFolderPath;
		}
		else if (type==1) // test dataset
		{
			negPosImagesNames.push_back(testNegImages);
			negPosImagesNames.push_back(testPosImages);
			posImgFolderPath = testPosImgFolderPath;
			negImgFolderPath = testNegImgFolderPath;
		}
		for(int j=0; j<2; j++)
		{
			stride = j ? posStride : negStride;  //take the negative stride when negative imaged and
			//the positive when positive
			for(int i=0; i<negPosImagesNames[j].size(); i++)
			{
				negImageCount=0;
				posCounter=0, negCounter=0;
				image = cv::imread(negPosImagesNames[j][i], CV_LOAD_IMAGE_GRAYSCALE);
				if(image.empty())	// skip if not an image
					continue;
				dotPosition = negPosImagesNames[j][i].find_last_of(".");  //find the position of the last dot in image name
				imageName	= negPosImagesNames[j][i].substr(0, dotPosition); // take the first part (before the extension dot)
				extension	= negPosImagesNames[j][i].substr(dotPosition);  //take the second part ( from the dot till the end)

				lastBackSlash = negPosImagesNames[j][i].find("\\"); //for some reason glob replaces the last "/" by '\\'
				maskAbsPath = masksPath + "/"+ imageName.substr(lastBackSlash+1) + ".mask.png" ;
				maskImg = cv::imread(maskAbsPath, CV_LOAD_IMAGE_UNCHANGED);

				std::cout<<i<<" "<<imageName.substr(lastBackSlash+1)<<"\n";
				for(int y=10; y< image.rows - 454 ; y+= stride)  //scan image rows
				{
					for(int x=10; x < image.cols - 454; x+=stride)  //scan image columns
					{

						if(cv::countNonZero(maskImg(cv::Rect(x,y,454,454))) == 0) // window outside the breast
						{
							x+= 454 - stride;	//make a big step, do not worry about -stride, +stride is coming next iteration :)
							continue;
						}
						if(cv::countNonZero(maskImg(cv::Rect(x,y,454,454))) == windowSize) //all pixels breast
						{
							window = image(cv::Rect(x, y, 454, 454)).clone();
							writeFileName = "/" + imageName.substr(lastBackSlash+1);
							if(j!=0) // it is an image the contains mass(es)
							{
								groundTruthImg = cv::imread(groundTruthPath + "/" + imageName.substr(lastBackSlash+1) + extension, CV_LOAD_IMAGE_UNCHANGED);
								GroundTruthRect = groundTruthImg(cv::Rect(x, y, 454, 454));
								windowCenter = GroundTruthRect(cv::Rect(220,220,15,15));	//rectangle 15 X 15 on ground truth window to check mass centering

								if(cv::countNonZero(GroundTruthRect) == 0)  //mass outside window
									{
										negImageCount++;
										if(! (negImageCount % stridesRatio))// take one negative window from a set
										{
											cv::imwrite(negImgFolderPath+ writeFileName + "_"+ std::to_string(i)+ "_n_"+std::to_string(++negCounter)+ ".tif", window);
											std::cout<<"count negative " <<i<<"\t"<<negCounter<<"\n";
										}
									}
								if(cv::countNonZero(windowCenter) == windowCenter.rows * windowCenter.cols ) //center all white ==> mass centered
									{
										cv::imwrite(posImgFolderPath + writeFileName +"_"+ std::to_string(i)+"_p_" +std::to_string(++posCounter)+ ".tif", window);
										std::cout<<"count positive " <<i<<"\t"<<posCounter<<"\n";
									}
							}
							else //j=0 ==> negative images
							{
								cv::imwrite(negImgFolderPath+ writeFileName + "_"+ std::to_string(i)+ "_n_"+std::to_string(++negCounter)+ ".tif", window);
										std::cout<<"count negative " <<i<<"\t"<<negCounter<<"\n";
							}


						}
					}
				}
			}
		}
	}
}


std::vector<std::vector<std::string>> aia::project0::splitNegPos(const std::vector<std::string> &images, const std::vector<std::string> &groundTruth)
{
	std::vector<std::string> posFiles;
	std::vector<std::string> negFiles;
	std::vector<std::vector<std::string>> dataBase;
	std::string pureName;	//the name of the image
	bool posImage= false;
	for(int i=0; i< images.size(); i++)
	{
		pureName = images[i].substr(images[i].find("\\")+1);
		posImage= false;	//reset
		for(int j=0; j<groundTruth.size() ; j++)
		{

			if(groundTruth[j].find(pureName) != std::string::npos )
			{
					posFiles.push_back(images[i]);
					posImage = true;
			}

		}
		if(!posImage)
			negFiles.push_back(images[i]);

	}
	dataBase.push_back(negFiles);
	dataBase.push_back(posFiles);
	return dataBase;
}


void aia::project0::reduceDataBase(std::string srcFolderPath, std::string dstFolderPath, std::string pattern, int samplesNum)
{
	std::vector<std::vector<std::string>> reducedDataBase;
	std::vector<std::string> srcNames, dstNames, selectedNames;
	float samplesCounter=0;

	cv::glob(srcFolderPath, srcNames);
	bool * checked = new bool [srcNames.size()];
	for(int i=0; i< srcNames.size(); i++)
		checked[i] = false;

	std::string imageName;
	for(int i=0; i<srcNames.size() ; i++)
	{
		if(checked[i])
			continue;

		imageName = srcNames[i].substr(0,srcNames[i].find_first_of(pattern));
		for(int j=0; j<srcNames.size() ; j++)
		{
			if(srcNames[j].find(imageName) != std::string::npos )
			{
				dstNames.push_back(srcNames[j]);
				checked[j] =true;

			}
		}
		if(dstNames.size() > samplesNum)
			{
				samplesCounter=0;
				while(samplesCounter < samplesNum && samplesCounter < dstNames.size())
				{
					selectedNames.push_back(dstNames[ceil((samplesCounter/samplesNum)* dstNames.size())]);
					samplesCounter++;

				}
				reducedDataBase.push_back(selectedNames);
				selectedNames.clear();

			}
			else
				reducedDataBase.push_back(dstNames);

			dstNames.clear();
	}
	for(int i=0; i<reducedDataBase.size(); i++)
		for(int j=0; j<reducedDataBase[i].size(); j++)
		{
			cv::imwrite(dstFolderPath + "/" + reducedDataBase[i][j].substr(reducedDataBase[i][j].find_last_of("\\")+1),
			cv::imread(reducedDataBase[i][j], CV_LOAD_IMAGE_UNCHANGED));

			std::cout<<" saving image i= "<< i<<"\t window j= " << j <<"\n";

		}


}
