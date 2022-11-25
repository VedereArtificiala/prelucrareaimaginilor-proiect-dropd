#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace sl;




class BackGroundExtractor
{
	//Using Running Gaussian average method on a grayscale image
	double p = 0.01;
	double tresh = 2.5;


	int w, h;

	float* mean, * disp;
	uchar* mask;


public:



	BackGroundExtractor(uchar* img, int width, int heigth) : w(width), h(heigth)
	{
		mean = new float[w * h];
		disp = new float[w * h];
		mask = new uchar[w * h];

		//memset mask as all background pixels
		std::memset(mask, 0, w * h * sizeof(uchar));
		//memset disp to some default
		std::memset(disp, 0, w * h * sizeof(float));
		//memcpy img to med

		for (int i = 0; i < w * h; ++i)
		{
			mean[i] = (float)img[i];
		}

	}

	~BackGroundExtractor()
	{
		delete mean;
		delete disp;
		delete mask;
	}


	void extract(uchar* img)
	{

		for (int i = 0; i < w * h; i++)
		{
			////If pixel is a background
			//if (mask[i] == 0)
			//{
			//	//updating the mean 
			//	mean[i] = p * img[i] + (1 - p) * mean[i];
			//}

			////updating the variance
			double d = abs(img[i] - mean[i]);
			//disp[i] = d * d * p + (1 - p) * disp[i];


			double sd = (sqrt(disp[i]));

			if ((d / sd) > tresh)
			{
				mask[i] = 255;
			}
			else
			{
				mask[i] = 0;
			}
		}

		//lets hope the sizeof(char) == sizeof(bool) on your machine
		std::memcpy(img, mask, w * h * sizeof(uchar));
	}


	void update(uchar* img)
	{
		for (int i = 0; i < w * h; i++)
		{
			//If pixel is a background
			if (mask[i] == 0)
			{
				//updating the mean 
				mean[i] = p * img[i] + (1 - p) * mean[i];
			}

			//updating the variance
			double d = abs(img[i] - mean[i]);
			disp[i] = d * d * p + (1 - p) * disp[i];
		}
	};

	void setMask(uchar* mask)
	{
		this->mask = mask;
	}
};





int getOCVtype(sl::MAT_TYPE type) {
	int cv_type = -1;
	switch (type) {
	case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
	case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
	case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
	case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
	case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
	case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
	case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
	case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
	default: break;
	}
	return cv_type;
}

//Conversion from Zed Mat to cv MAT
cv::Mat slMat2cvMat(Mat& input) {
	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	// cv::Mat and sl::Mat will share a single memory structure
	return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}



int main(int argc, char** argv) {

	//// Create a ZED camera object
	Camera zed;

	//// Set configuration parameters
	//InitParameters init_parameters;
	//init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE; // Use PERFORMANCE depth mode
	//init_parameters.coordinate_units = UNIT::MILLIMETER; // Use millimeter units (for depth measurements)

	//// Open the camera
	//auto returned_state = zed.open(init_parameters);
	//if (returned_state != ERROR_CODE::SUCCESS) {
	//    cout << "Error " << returned_state << ", exit program." << endl;
	//    return EXIT_FAILURE;
	//}

	//// Set runtime parameters after opening the camera
	RuntimeParameters runtime_parameters;
	runtime_parameters.sensing_mode = SENSING_MODE::STANDARD; // Use STANDARD sensing mode


	int i = 0;
	sl::Mat zedImage, tmp;
	cv::Mat image, imgO, imgProc, background;
	vector<cv::Vec4i> linesP;

	cv::VideoCapture cap = cv::VideoCapture(0);
	if (!cap.isOpened())
		return 0;


	cap.read(image);
	cv::cvtColor(image, imgO, cv::COLOR_RGB2GRAY);
	BackGroundExtractor bg(imgO.data, imgO.size().width, imgO.size().height);

	int k = 0;
	//while (1)
	//{

	//	cap.read(image);
	//	cv::cvtColor(image, imgO, cv::COLOR_RGB2GRAY);
	//	if (k == 20)
	//	{

	//		k = 0;
	//	}
	//	bg.update(imgO.data);
	//	bg.extract(imgO.data);
	//	cv::erode(imgO, imgO, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1)), cv::Point(-1, -1), 2, 0, cv::morphologyDefaultBorderValue());
	//	cv::dilate(imgO, imgO, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1)), cv::Point(-1, -1), 2, 0, cv::morphologyDefaultBorderValue());
	//	//bg.setMask(imgO.data);

	//	cv::imshow("", imgO);
	//	cv::waitKey(1);
	//	k++;
	//}

	cv::Ptr<cv::BackgroundSubtractor> b = cv::createBackgroundSubtractorMOG2(200, 20, false);

	for (int i = 0; i < 200; i++)
	{
		cap.read(image);
		b->apply(image, imgO);
	}

	while (1)
	{
		//read image and calculate foreground mask
		cap.read(image);

		//cv::imshow("", image);
		//cv::waitKey();

		b->apply(image, imgO, 0);
		//apply morphologic operations to reduce noise and minimize false-negative cases
		cv::erode(imgO, imgO, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1)), cv::Point(-1, -1), 1, 0, cv::morphologyDefaultBorderValue());
		cv::dilate(imgO, imgO, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1)), cv::Point(-1, -1), 10, 0, cv::morphologyDefaultBorderValue());
		
		//cv::imshow("", imgO);
		//cv::waitKey();

		//apply foreground mask
		cv::cvtColor(image, image, cv::COLOR_RGB2GRAY, 0);
		for (int i = 0; i < imgO.size().width * imgO.size().height; i++) 
		{
			if (imgO.at<char>(i) == 0)
				image.at<char>(i) = 0;
		}

		//cv::imshow("", image);
		//cv::waitKey();

		//calculate threshold through Otsu's method, apply Canny and HoughP
		double temp = cv::threshold(image, imgO, 0, 255, cv::THRESH_OTSU);
		cv::Canny(image, imgO, (int)(temp / 5), (int)temp, 3, false);

		//cv::imshow("", imgO);
		//cv::waitKey();

		cv::HoughLinesP(imgO, linesP, 1, CV_PI / 180, 190, 180, 50);

		for (size_t i = 0; i < linesP.size(); i++)
		{
			cv::Vec4i l = linesP[i];
			cv::line(image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
		}

		cv::imshow("", image);
		cv::waitKey(1);
	}

	//zed.grab(runtime_parameters);
	//zed.retrieveImage(zedImage, VIEW::LEFT_GRAY);
	//cv::imshow("dfhdfgjdhjdhgj", slMat2cvMat(zedImage));
	//cv::waitKey();
	//zed.grab(runtime_parameters);
	//zed.retrieveImage(zedImage, VIEW::LEFT_GRAY);
	//zedImage.copyTo(tmp);
	//background = slMat2cvMat(tmp);

	while (true) {
		// A new image is available if grab() returns ERROR_CODE::SUCCESS
		if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {
			// Retrieve left image
			zed.retrieveImage(zedImage, VIEW::LEFT_GRAY);
			image = slMat2cvMat(zedImage);

			for (int i = 0; i < image.size().area(); i++)
			{
				if (abs(image.at<char>(i) - background.at<char>(i)) < 30)
					image.at<char>(i) = 0;
				else {}
			}

			image.copyTo(imgO);

			//
			double temp = cv::threshold(image, imgProc, 0, 255, cv::THRESH_OTSU);

			cv::Canny(image, imgProc, (int)(temp / 5), (int)temp, 3, false);
			cv::HoughLinesP(imgProc, linesP, 1, CV_PI / 180, 250, 200, 60);

			for (size_t i = 0; i < linesP.size(); i++)
			{
				cv::Vec4i l = linesP[i];
				cv::line(imgO, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
			}

			cv::imshow("houghP", image);
			cv::waitKey();
		}
	}

	//Close the camera
	zed.close();
	return EXIT_SUCCESS;
}




