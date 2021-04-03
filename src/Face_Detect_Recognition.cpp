// Face_Detect_Recognition.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>


// 快速宣告的方法 : 
using namespace cv;
using namespace std; 
using namespace cv::dnn;


const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);
float confidenceThreshold = 0.9;

//cos 相似性度量
float cos_distance(vector<float> vecfeature1, vector<float> vecfeature2)
{
	float cos_dis = 0;
	float dotmal = 0, norm1 = 0, norm2 = 0;

	for (int i = 0; i < vecfeature1.size(); i++)
	{
		// cout << "vecfeature1-after : " << vecfeature1[i] << endl;
		// cout << "vecfeature2-after : " << vecfeature2[i] << endl;
		dotmal += vecfeature1[i] * vecfeature2[i];
		norm1 += vecfeature1[i] * vecfeature1[i];
		norm2 += vecfeature2[i] * vecfeature2[i];
	}
	// cout << " dtotmal : " << dotmal << endl;
	norm1 = sqrt(norm1);
	// cout << " norm1 : " << norm1 << endl;
	norm2 = sqrt(norm2);
	// cout << " norm2 : " << norm2 << endl;
	cos_dis = dotmal / (norm1 * norm2);
	return cos_dis;
}


float Euc_Dis(vector<float> v1, vector<float> v2)
{
	float euclidean = 0, euclidean2 = 0, x1 = 0, y1 = 0;
	for (int i = 0; i < v1.size(); i++)
		x1 += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	// cout << x1 << endl;
	euclidean = sqrt(x1);
	// cout << euclidean << endl;

	return euclidean;
}

int main(int argc, char** argv)
{
	// Step1.Load Data(image, prototxt, caffemodel)

	//  A.Load prototxt, caffemodel, image : 
    //		A1.Detection Model : 
	std::string Detection_Prototxt = "./data/model/deploy/yufacedetectnet.prototxt"; // "./data/caffe_network/deploy_txt/hubery_deploy.prototxt";
	std::string Detection_Caffemodel = "./data/model/caffemodel/yufacedetectnet.caffemodel"; // "./data/caffe_network/caffemodel/0115_Add_PRelu_Blob2.caffemodel";
	dnn::Net Detection_net = readNetFromCaffe(Detection_Prototxt, Detection_Caffemodel);
	//		A2 Varify : 驗證是否讀取Model成功 :

	if (Detection_net.empty() == false)
	{
		cout << "Detection Model (caffemodel & prototxt) Loading success !" << endl;
	}
	else
	{
		cout << "FAIL!!!" << endl;
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt :   " << Detection_Prototxt << endl;
		cerr << "caffemodel : " << Detection_Caffemodel << endl;
		cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		exit(-1);
	}

	//		B1.Recognition Model : 
	std::string caffemodel_path = "./data/model/caffemodel/";
	std::string deploy_path = "./data/model/deploy/";
	std::string caffemodel = caffemodel_path + "CaffeNet.caffemodel";      // "Tiny_VGGFace.caffemodel" "CaffeNet.caffemodel"
	std::string caffe_prototxt = deploy_path + "CaffeNet_deploy.prototxt"; // "Tiny_VGGFace.prototxt" "CaffeNet_deploy.prototxt"
	cv::dnn::Net Recognition_net = readNetFromCaffe(caffe_prototxt, caffemodel);
	//		B2.Varify : 驗證是否讀取Model成功 :
	if (Recognition_net.empty() == false)
	{
		cout << "Recognition Model (caffemodel & prototxt) Loading success !" << endl;
	}
	else
	{
		cout << "FAIL!!!" << endl;
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt :   " << Detection_Prototxt << endl;
		cerr << "caffemodel : " << Detection_Caffemodel << endl;
		cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		exit(-1);
	}

	//		C1.Image : 
	string inputpath = "./data/image/person/"; // set input path // "./data/image/face/"
	String imageFile = "workout.jpg"; // "Ben.jpg" david_fun.jpg
	String imageFile2 = "david.jpg"; // "david.jpg" "tom.jpg" "black.jpg" han.JPG "david_hope.JPG"

	Mat img = imread(inputpath + imageFile);
	Mat img2 = imread(inputpath + imageFile2);

	cout << "Width : " << img.size().width << endl;
	cout << "Height: " << img.size().height << endl;
	int w = img.size().width;
	int h = img.size().height;
	string outputpath = "./data/output/"; // set output path
	string FinishedDetectImg = outputpath + "roi.jpg";
	cout << "FinishedDetectImg :" << FinishedDetectImg << endl;

	//		C2.Varify : 驗證是否讀取Image成功 :
	if (img.empty() == true)
	{
		cout << "Can't read image" << endl;
	}
	else
	{
		cout << "Read image : Sucess !" << endl;
	}

	if (img2.empty() == true)
	{
		cout << "Can't read image2" << endl;
	}
	else
	{
		cout << "Read image2 : Sucess !" << endl;
	}

	// Step2.Face Detection : 

	//		2.1 圖片前處理
	cv::Scalar mean(104, 117, 123, 0);
	cv::Mat blob;
	cv::dnn::blobFromImage(img, blob);
	Detection_net.setInput(blob, "data");
	cv::Mat prob = Detection_net.forward();

	//		2.2 Detection face by ROI which's higher then threshold :
	cv::Mat detectionMat(prob.size[2], prob.size[3], CV_32F, prob.ptr<float>());
	printf(" confidenceThreshold = %f\n ", confidenceThreshold);
	printf("\n ");

	// 2.3 Show the  detected BBOX-data : 
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

			int x_offset = (xLeftBottom + xRightTop) / 2;
			printf("x_offset : ", x_offset);
			int y_offset = (yLeftBottom + yRightTop) / 2;
			printf("x_offset : ", x_offset);

			Rect rect(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
			Mat roi = Mat(img, rect);
			Mat imgRect = img.clone();
			imwrite(outputpath + imageFile, roi);
			resize(roi, roi, Size(96, 112));
			imwrite(outputpath +"roi.jpg", roi);

			cv::Rect object(
				(int)xLeftBottom,
				(int)yLeftBottom,
				(int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));

			cv::rectangle(img, object, cv::Scalar(0, 255, 0), 3);
			imwrite(outputpath + "ROI_BBOX.jpg", img); 
		}
	}


	// Step3.Face-Recognition : 

	Mat vecfeature1;
	Mat vecfeature2;
	
	//		3.1 Image1(Main-image) : 
	cout << " < 1.Main-image(image1) > " << endl;

	//			3.1.1 glob blob-data from caffemodel and preprocessing image : 
	Mat img1 = imread(FinishedDetectImg);
	//cv::resize(img, img, cv::Size(96, 112)); // already resize in step2 
	Mat inputBlob = dnn::blobFromImage(img1, 0.00390625f, Size(96, 112), Scalar(), false); //Convert Mat to batch of images
	Recognition_net.setInput(inputBlob, "data");  // The caffemodel's Input
	vecfeature1 = Recognition_net.forward("fc5");  // Turn to Vector(type), The last output in caffemodel
	//cout << " vecfeature1 : " << vecfeature1 << endl; // Show the last output value
	cout << " vecfeature1.size : " << vecfeature1.size() << endl; // Show image1's Vector-size

	//			3.1.2 Normlization and compute euclidean distance:
	vector<double> x = vecfeature1.clone();
	float euc = norm(x); // normalization input image;
	cout << " Image1 - Euclidean distance  : " << euc << endl;
	//			3.1.3 Compute L2 Norm's :
	vector<float> l2_norm;
	for (int i = 0; i < x.size(); i++)
	{
		l2_norm.push_back(x[i] / euc);
		// cout << "1.Dimension " << i + 1 << " : " << L2_norm[i] << endl; // Show each dimension's L2-norm value 
	}
	//cout << " vector1 : " << l2_norm.size() << endl; // Check dimension(512) correct or not
	cout << " " << endl; // 分隔image1 & image2

	//      3.2 Image2(compared image) : Do the same things with 3.1 :
	cout << " < 2.Compare-image(image2) >" << endl;
	cv::resize(img2, img2, cv::Size(96, 112));
	Mat inputBlob2 = dnn::blobFromImage(img2, 0.00390625f, Size(96, 112), Scalar(), false);
	Recognition_net.setInput(inputBlob2, "data");
	vecfeature2 = Recognition_net.forward("fc5");
	vector<float> y = vecfeature2.clone();
	// cout << "y :" << y.size() << endl; // check y size
	float euc2 = norm(y);
	cout << "2.image2's euclidean distance  : " << euc2 << endl;
	vector<float> l2_norm2;
	for (int j = 0; j < y.size(); j++)
	{
		l2_norm2.push_back(y[j] / euc2);
		// cout << " 2.Dimension2 " << j + 1 << " : " << l2_norm2[j] << endl;
	}
	cout << " vector2.size : " << l2_norm2.size() << endl;

	//		3.3 Calculate Cosine-Similarity/ Euclidean-Distance :  
	cout << "" << endl;
	cout << " < Image name > : " << imageFile2 << endl;

	//			3.3.1 Cosine-Similarity : cout << "" << endl;
	float cos_sim = cos_distance(l2_norm, l2_norm2);
	cout << " - Cosine-Similarity-Distance = " << cos_sim << endl;

	//			3.3.2 Euclidean-distance :
	float Euc = Euc_Dis(l2_norm, l2_norm2);
	cout << " - Euclidean-Distance = " << Euc << endl;

	//	3.4 Threshold setting :


	/**/

	return 0;
}


/* The part of Webcam : 
	// VideoCapture class for playing video for which faces to be detected
	VideoCapture capture;
	Mat frame, image;

	// PreDefined trained XML classifiers with facial features
	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	// Load classifiers from "opencv/data/haarcascades" directory 
	nestedCascade.load("./data/model/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

	// Change path before execution
	cascade.load("./data/model/haarcascades/haarcascade_frontalcatface.xml");

	if (cascade.empty() == true)
	{
		cout << "No frontalcatface.xml" << endl;
	}
	else
	{
		cout << "frontalcatface.xml load Sucess !!" << endl;
	}

	if (nestedCascade.empty() == true)
	{
		cout << "No eye_tree_eyeglasses.xml" << endl;
	}
	else
	{
		cout << "eye_tree_eyeglasses.xml load Sucess !!" << endl;
	}

	// Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video

	VideoCapture cap(0);

	if (capture.isOpened())
	{
		// Capture frames from video and detect faces
		cout << "Face Detection Started...." << endl;
		if (!cap.isOpened())
			cout << "Couldn't open camera : " << endl;
		return -1;
	}

	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera/video or read image

		if (frame.empty())
		{
			waitKey();
			break;
		}

		if (frame.channels() == 4)
			cvtColor(frame, frame, COLOR_BGRA2BGR);


		Mat inputBlob = blobFromImage(frame, inScaleFactor,
			Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images  
															 //! [Prepare blob]  

															 //! [Set input blob]  
		net.setInput(inputBlob, "data"); //set the network input  
										 //! [Set input blob]  

										 //! [Make forward pass]  
		Mat detection = net.forward("detection_out"); //compute output  
													  //! [Make forward pass]  

		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;

		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		ostringstream ss;
		ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
		putText(frame, ss.str(), Point(20, 20), 1, 1, Scalar(0, 0, 255));


		float confidenceThreshold = min_confidence;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > confidenceThreshold)
			{
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				rectangle(frame, object, Scalar(0, 255, 0));

				ss.str("");
				ss << confidence;
				String conf(ss.str());
				String label = "David: " + conf;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 0.5, &baseLine);
				rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), CV_FILLED);
				putText(frame, label, Point(xLeftBottom, yLeftBottom),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
		cv::imshow("Detections", frame);

		// Press q to exit from window
		char c = (char)waitKey(10);
		//if (waitKey(1) >= 0) break;
		if (c == 27 || c == 'q' || c == 'Q')
			break;

	}

	return 0;
}
*/

/*
		while (1)
		{
			if (frame.empty())
				break;
			imshow("AVER_Face_Detection", frame);
			Mat frame1 = frame.clone();
			//detectAndDraw(frame1, cascade, nestedCascade, scale);
			if (waitKey(60) > 0)
				break;
		}
		return 0;

		/*while (1)
		{
			capture >> frame;
			if (frame.empty())
				break;
			Mat frame1 = frame.clone();
			detectAndDraw(frame1, cascade, nestedCascade, scale);
			char c = (char)waitKey(10);

			// Press q to exit from window
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}
	else
		cout << "Could not Open Camera";
		*/
	//
/**/


	/*
		// Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
		capture.open(0);
		if (capture.isOpened())
		{
			// Capture frames from video and detect faces
			cout << "Face Detection Started...." << endl;
			while (1)
			{
				capture >> frame;
				if (frame.empty())
					break;
				Mat frame1 = frame.clone();
				detectAndDraw(frame1, cascade, nestedCascade, scale);
				char c = (char)waitKey(10);

				// Press q to exit from window
				if (c == 27 || c == 'q' || c == 'Q')
					break;
			}
		}
		else
			cout << "Could not Open Camera";
	*/




/*
int main()
{
	VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;

	Mat frame;
	while (1)
	{
		cap >> frame;
		imshow("Test", frame);
		if (waitKey(60) > 0)
			break;
	}
	return 0;
}
*/



/* Hello World : 

int main()
{
	std::cout << "Hello World!\n";
}

*/





// 執行程式: Ctrl + F5 或 [偵錯] > [啟動但不偵錯] 功能表
// 偵錯程式: F5 或 [偵錯] > [啟動偵錯] 功能表

