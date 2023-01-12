#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;


class BackGroundExtractor
{
	//Using Running Gaussian average method on a grayscale image
	double p = 0.5;
	double tresh = 2.5;

	int bufferSize;


	double* disp;
	double* mean;
	uchar* mask;


public:

	BackGroundExtractor(uchar* img, int width, int heigth) : bufferSize(width * heigth)
	{
		mean = new double[bufferSize];
		disp = new double[bufferSize];
		mask = new uchar[bufferSize];

		//memset mask all pixels as background
		std::memset(mask, 0, bufferSize * sizeof(uchar));
		//memset disp to some default
		std::memset(disp, 0, bufferSize * sizeof(double));
		//memcpy img to med

		for (int i = 0; i < bufferSize; ++i)
		{
			mean[i] = (double)img[i];
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
		if(p > 0.05)
			p = 0.95 * p;
		for (int i = 0; i < bufferSize; i++)
		{
			double distance = abs((double)img[i] - mean[i]);

			if (mask[i] == 0)
			{
				//updating the mean 
				mean[i] = (1 - p) * mean[i] + p * (double)img[i];
				//updating the variance
				disp[i] = (1 - p) * disp[i] + distance * distance * p;
			}

			if ((distance / sqrt(disp[i])) > tresh)
			{
				mask[i] = 255;
			}
			else
			{
				mask[i] = 0;
			}

			//cout << "(" << distance << ", " << disp[i] << ", " << (int)mask[i] << " )\n";
		}

		std::memcpy(img, mask, bufferSize * sizeof(uchar));

	}


	void update(uchar* img)
	{
		for (int i = 0; i < bufferSize; i++)
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


struct LinesCluster
{
	cv::Point2i minx = {720, 0}, maxx = { 0, 0 }, miny = { 0, 1280 }, maxy = { 0, 0 };
	std::vector<cv::Vec4i> lines;
	double avgOrientation;
	std::vector<double> angles;
	double angleSum;
};

//Angle in radians of a line with x axis
double getLineAngle(cv::Vec4i l)
{
	if (l[0] == l[2])
		return 1.5708; //~90 degrees in radians
	else if (l[1] == l[3])
		return 0.; //0 degrees
	else
	{
		//make l[0] always min(x, x2)
		if (l[0] > l[2])
		{
			int tmp = l[0];
			l[0] = l[2];
			l[2] = tmp;
			tmp = l[1];
			l[1] = l[3];
			l[3] = l[1];
		}

		double ang = abs(l[1] - l[3]) / sqrt((l[0] - l[2]) * (l[0] - l[2]) + (l[1] - l[3]) * (l[1] - l[3]));
		//return (l[1] < l[3]) ? M_PI - ang : ang;
		return ang;
	}
}

LinesCluster getLargestCluster(std::vector<cv::Vec4i> linesP)
{
	std::vector<LinesCluster> clusters;

	cv::Vec4i l = linesP[0];
	double lineAngle = getLineAngle(l);

	// create cluster with first line
	LinesCluster tmp;
	tmp.lines.push_back(l);
	tmp.angles.push_back(lineAngle);
	tmp.angleSum = lineAngle;
	tmp.avgOrientation = lineAngle;

	clusters.push_back(tmp);

	for (int i = 1; i < linesP.size(); i++)
	{
		l = linesP[i];
		lineAngle = getLineAngle(l);

		bool found = false;
		for (int k = 0; k < clusters.size(); k++) // searching a cluster with average angle = line angle +- 5 degrees
		{
			if (abs(lineAngle - clusters[k].avgOrientation) < 0.087) // ~5 degrees
			{
				cv::Vec4i testLine = clusters[k].lines[0];
				if (abs(l[1] - testLine[1]) < 100) //distance between lines less than 100px 
				{
					clusters[k].lines.push_back(l);
					clusters[k].angles.push_back(lineAngle);
					clusters[k].angleSum += lineAngle;
					clusters[k].avgOrientation = clusters[k].angleSum / (clusters[k].angles.size());

					found = true;
					break;
				}
			}
		}
		if (!found) // create a new cluster
		{
			tmp.lines.clear();
			tmp.lines.push_back(l);
			tmp.angles.clear();
			tmp.angles.push_back(lineAngle);
			tmp.angleSum = lineAngle;
			tmp.avgOrientation = lineAngle;

			clusters.push_back(tmp);
		};
	}

	tmp = clusters[0];
	for (int i = 0; i < clusters.size(); i++) // find the largest one
		if (clusters[i].lines.size() > tmp.lines.size())
			tmp = clusters[i];

	//finding cluster extreme points
	for (int i = 0; i < tmp.lines.size(); i++) 
	{
		l = tmp.lines[i];
		if (l[0] < tmp.minx.x)
			tmp.minx = { l[0], l[1] };
		
		if(l[2] > tmp.maxx.x)
			tmp.maxx = { l[2], l[3] };

		if (l[1] < l[3])
		{
			if (l[1] < tmp.miny.y)
				tmp.miny = { l[0], l[1] };

			if(l[3] > tmp.maxy.y)
				tmp.maxy = { l[2], l[3] };
		}
		else
		{
			if (l[3] < tmp.miny.y)
				tmp.miny = { l[2], l[3] };

			if (l[1] > tmp.maxy.y)
				tmp.maxy = { l[0], l[1] };
		}
	}

	l = tmp.lines[0];
	if (l[1] < l[3])
		tmp.angleSum *= -1;

	return tmp;
}

cv::Mat rotateAndCrop(LinesCluster cluster, cv::Mat img)
{
	cv::Vec4i line = cluster.lines[0];

	//center of rotation
	cv::Point2f center = { (float)(cluster.minx.x + (cluster.maxx.x - cluster.minx.x) / 2), (float)(cluster.miny.y + (cluster.maxy.y - cluster.miny.y) / 2)};

	//rotate corners
	cv::Point2i c[4] = { cluster.maxx, cluster.minx , cluster.maxy , cluster.miny };
	cluster.avgOrientation = (cluster.angleSum / cluster.angles.size());
	double cosa = cos(cluster.avgOrientation), sina = sin(cluster.avgOrientation);
	for (int i = 0; i < 4; i++)
	{

		c[i].x -= center.x;
		c[i].y -= center.y;

		c[i].x = c[i].x * cosa - c[i].y * sina;
		c[i].y = c[i].x * sina + c[i].y * cosa;

		c[i].x += center.x;
		c[i].y += center.y;
	}

	c[3].y -= 20;
	c[2].y += 20;

	// rotate image
	cv::Mat rotationMat = cv::getRotationMatrix2D(center, cluster.avgOrientation * -180 / CV_PI, 1);
	cv::Mat rotatedImage;
	cv::warpAffine(img, rotatedImage, rotationMat, { img.cols, img.rows }, cv::INTER_NEAREST);

	if (c[0].x > rotatedImage.cols)
		c[0].x = rotatedImage.cols;
	if (c[2].y > rotatedImage.rows)
		c[2].y = rotatedImage.rows;
	if (c[1].x < 0)
		c[1].x = 0;
	if (c[3].y < 0)
		c[3].y = 0;

	//if (cluster.avgOrientation >= 0.6)
	//{
	//	c[3].y -= 40;
	//	c[2].y += 40;
	//}
	//else if (cluster.avgOrientation >= 0)
	//	c[3].y -= 20;

	if (c[2].y < c[3].y)
	{
		int tmp = c[2].y;
		c[2].y = c[3].y;
		c[3].y = tmp;
	}

	//maybe a bug, lets keep it like this for now
	if (cluster.avgOrientation >= 0 || (c[2].y - c[3].y) < 100)
	{
		c[3].y -= 50;
		c[2].y += 30;
	}
 
	try
	{
		cv::Mat neck(rotatedImage, cv::Rect(cv::Point(c[1].x, c[3].y - 10), cv::Point(c[0].x, c[2].y + 10)));
		return neck;
	}
	catch (exception)
	{
		return rotatedImage;
	}
}




int main(int argc, char** argv) {


	cv::Mat image, imgO, imgProc, background;
	vector<cv::Vec4i> linesP;

	cv::VideoCapture cap = cv::VideoCapture(0);
	if (!cap.isOpened())
		return 0;

	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);


	cap.read(image);
	cv::cvtColor(image, imgO, cv::COLOR_RGB2GRAY);
	//cv::normalize(imgO, imgO, 0, 10000, cv::NORM_MINMAX, CV_8UC1);


	//Running Gaussian Average
	//BackGroundExtractor bg(imgO.data, imgO.size().width, imgO.size().height);
	//while (1)
	//{
	//	cap.read(image);
	//	cv::cvtColor(image, imgO, cv::COLOR_RGB2GRAY);
	//	//bg.update(imgO.data);
	//	//cv::normalize(imgO, imgO, 0, 10000, cv::NORM_MINMAX, CV_8UC1);
	//	bg.extract(imgO.data);
	//	//cv::erode(imgO, imgO, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1)), cv::Point(-1, -1), 2, 0, cv::morphologyDefaultBorderValue());
	//	//cv::dilate(imgO, imgO, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1)), cv::Point(-1, -1), 2, 0, cv::morphologyDefaultBorderValue());
	//	//bg.setMask(imgO.data);

	//	cv::imshow("", imgO);
	//	cv::waitKey(1);
	//}

	//Using MOG2 for that demo
	cv::Ptr<cv::BackgroundSubtractor> b = cv::createBackgroundSubtractorMOG2(100, 20, false);
	for (int i = 0; i < 100; i++)
	{
		cap.read(image);
		b->apply(image, imgO);
	}

	while (1)
	{
		//read image
		cap.read(image);

		//calculate foreground mask
		b->apply(image, imgO, 0);

		//apply morphologic operations to reduce noise and minimize false-negative cases
		cv::erode(imgO, imgO, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1)), cv::Point(-1, -1), 1, 0, cv::morphologyDefaultBorderValue());
		cv::dilate(imgO, imgO, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1)), cv::Point(-1, -1), 10, 0, cv::morphologyDefaultBorderValue());

		//apply foreground mask
		cv::cvtColor(image, image, cv::COLOR_RGB2GRAY, 0);
		for (int i = 0; i < imgO.size().width * imgO.size().height; i++) 
		{
			if (imgO.at<char>(i) == 0)
				image.at<char>(i) = 0;
		}


		//calculate threshold through Otsu's method, apply Canny
		double temp = cv::threshold(image, imgO, 0, 255, cv::THRESH_OTSU);
		cv::Canny(image, imgO, (int)(temp / 3), (int)temp * 1.3, 3, false);


		//apply HoughP transform
		cv::HoughLinesP(imgO, linesP, 1, CV_PI / 180, 190, 180, 50);

		//if (linesP.size() != 0)
		//{
		//	LinesCluster tst = getLargestCluster(linesP);

		//	for (size_t i = 0; i < tst.lines.size(); i++)
		//	{
		//		cv::Vec4i l = tst.lines[i];
		//		cv::line(image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
		//	}
		//}

		//if there are lines, find the cluster
		if (linesP.size() != 0) 
		{
			LinesCluster bassLine = getLargestCluster(linesP);

			if(bassLine.lines.size() > 5)
				imgO = rotateAndCrop(bassLine, image);
		}

		//If image was cropped, we supposed the neck was detected
		if (imgO.rows != 720 && imgO.cols != 1280)
		{
			cv::Sobel(imgO, imgO, -1, 1, 0, 3);
		}

		cv::imshow("", imgO);
		cv::waitKey(1);

	}

	return EXIT_SUCCESS;
}




