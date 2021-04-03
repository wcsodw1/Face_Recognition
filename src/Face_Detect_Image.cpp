// Face_Detect_Recognition.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>

// 快速宣告的方法 : 
using namespace cv;
using namespace std; 
using namespace cv::dnn;


// Function for Face Detection
//void detectAndDraw( Mat& img, CascadeClassifier& cascade, 
//                CascadeClassifier& nestedCascade, double scale );string cascadeName, nestedCascadeName;

/*

int main(int argc, char** argv)
{
	// 1.Load Data(image, prototxt, caffemodel)

		// a.Load prototxt, caffemodel, image
	cout << "Hello world!" << endl;
	std::string Caffe_Prototxt = "./data/model/deploy/yufacedetectnet.prototxt"; // "./data/caffe_network/deploy_txt/hubery_deploy.prototxt";
	std::string Caffemodel = "./data/model/caffemodel/yufacedetectnet.caffemodel"; // "./data/caffe_network/caffemodel/0115_Add_PRelu_Blob2.caffemodel";
	string inputpath = "./data/image/group/"; // set input path 
	String imageFile = "prolongon.jpg";
	Mat img = imread(inputpath + imageFile);
	string outputpath = "./data/output/"; // set output path 

	// VideoCapture class for playing video for which faces to be detected
	VideoCapture capture;
	Mat frame, image;


		// b.DNN model load - 載入網絡 
	dnn::Net net = readNetFromCaffe(Caffe_Prototxt, Caffemodel);

	   // c.Varify : 驗證有無抓到 Net/image Log :
		if (net.empty() == false)
	{
		cout << "Load model (caffemodel & prototxt) success!" << endl;
	}

	else
	{
		cout << "FAIL!!!" << endl;
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt :   " << Caffe_Prototxt << endl;
		cerr << "caffemodel : " << Caffemodel << endl;
		cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		exit(-1);
	}

    if (img.empty() == true)
    {
        cout <<"Can't read image"<< endl;
    }

    else
    {
        cout << "Read image : Sucess !!" << endl;
    }

	// 2.圖片前處理
	cv::Scalar mean(104, 117, 123, 0);
	cv::Mat blob;
	cv::dnn::blobFromImage(img, blob);
	net.setInput(blob, "data");
	cv::Mat prob = net.forward();

	// 3.Select the one who higher then threshold : 
	cv::Mat detectionMat(prob.size[2], prob.size[3], CV_32F, prob.ptr<float>());
	float confidenceThreshold = 0.5;
	// printf(" confidenceThreshold = %f\n ", confidenceThreshold);
	printf("\n ");

	for (int i = 0; i < detectionMat.rows; i++)
	{

		float confidence = detectionMat.at<float>(i, 2);
		printf("confidence = %f\n ", confidence);

		if (confidence > confidenceThreshold)
		{
			// 高於置信度的，獲取其x、y、以及對應的寬度高度，進行框選
			int classId = (detectionMat.at<float>(i, 1));
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
			printf("xLeftBottom : %i\n ", xLeftBottom);

			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
			printf("yLeftBottom : %i\n ", yLeftBottom);

			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
			printf("xRightTop   : %i\n ", xRightTop);

			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);
			printf("yRightTop   : %i\n ", yRightTop);

			// int twoC = static_cast<int>(detectionMat.at<float>(i, 2) * img.cols);
			// printf("two cols   : %i\n ", twoC);

			// int twoR = static_cast<int>(detectionMat.at<float>(i, 2) * img.rows);
			// printf("two rows   : %i\n ", twoR);

			printf("\n ");

			cv::Rect object((int)xLeftBottom,
				(int)yLeftBottom,
				(int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));

			cv::rectangle(img, object, cv::Scalar(0, 255, 0), 2);
			// qDebug() << __FILE__ << __LINE__
			//         << classId
			//         << confidence << confidenceThreshold
			//         << object.x << object.y << object.width << object.height;
		}

	}

	cv::imwrite(outputpath + "ProLonGon.jpg", img);

	return 0;
}

*/

