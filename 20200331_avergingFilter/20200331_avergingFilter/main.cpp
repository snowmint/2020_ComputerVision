#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <windows.h>
#include <math.h>
#include <random>

using namespace std;
using namespace cv;
double alpha; /**< Simple contrast control */
int beta;  /**< Simple brightness control */


int computeMedian(vector<int> elements)
{
	nth_element(elements.begin(), elements.begin() + elements.size() / 2, elements.end());

	//sort(elements.begin(),elements.end());
	return elements[elements.size() / 2];
}

Mat compute_median(std::vector<cv::Mat> vec) {
	// Note: Expects the image to be CV_8UC3
	Mat medianImg(vec[0].rows, vec[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int row = 0; row<vec[0].rows; row++) {
		for (int col = 0; col<vec[0].cols; col++) {
			vector<int> elements_B;
			vector<int> elements_G;
			vector<int> elements_R;

			for (int imgNumber = 0; imgNumber<vec.size(); imgNumber++)
			{
				/*int B = vec[imgNumber].at<cv::Vec3b>(row, col)[0];
				int G = vec[imgNumber].at<cv::Vec3b>(row, col)[1];
				int R = vec[imgNumber].at<cv::Vec3b>(row, col)[2];*/
				int B = vec[imgNumber].ptr<Vec3b>(row)[col].val[0];
				int G = vec[imgNumber].ptr<Vec3b>(row)[col].val[1];
				int R = vec[imgNumber].ptr<Vec3b>(row)[col].val[2];

				elements_B.push_back(B);
				elements_G.push_back(G);
				elements_R.push_back(R);
			}
			/*medianImg.at<cv::Vec3b>(row, col)[0] = computeMedian(elements_B);
			medianImg.at<cv::Vec3b>(row, col)[1] = computeMedian(elements_G);
			medianImg.at<cv::Vec3b>(row, col)[2] = computeMedian(elements_R);*/
			medianImg.ptr<Vec3b>(row)[col].val[0] = computeMedian(elements_B);
			medianImg.ptr<Vec3b>(row)[col].val[1] = computeMedian(elements_G);
			medianImg.ptr<Vec3b>(row)[col].val[2] = computeMedian(elements_R);
		}
		cout << "keep waiting" << row << endl;
	}
	return medianImg;
}

int main() {
	VideoCapture capture("demo2.mp4");

	if (!capture.isOpened())
		cerr << "Error opening video file\n";
	default_random_engine generator;
	uniform_int_distribution<int>distribution(0, capture.get(CAP_PROP_FRAME_COUNT));

	vector<Mat> frames;
	Mat frame;

	for (int i = 0; i<5; i++)
	{
		int fid = distribution(generator);
		capture.set(CAP_PROP_POS_FRAMES, fid);
		Mat frame;
		capture >> frame;
		if (frame.empty())
			continue;
		frames.push_back(frame);
		cout << "keep" << i << endl;
	}
	// Calculate the median along the time axis
	cout << "caculating..." << endl;
	Mat medianFrame = compute_median(frames);
	cout << "result" << endl;
	// Display median frame
	imshow("background image", medianFrame);
 

	capture.set(CAP_PROP_POS_FRAMES, 0);
	Mat grayMedianFrame;
	cvtColor(medianFrame, grayMedianFrame, COLOR_BGR2GRAY);

	// Loop over all frames
	while (1)
	{
		// Read frame
		capture >> frame;

		if (frame.empty())
		{
			break;
		}

		// Convert current frame to grayscale
		cvtColor(frame, frame, COLOR_BGR2GRAY);

		// Calculate absolute difference of current frame and the median frame
		Mat dframe;
		absdiff(frame, grayMedianFrame, dframe);

		// Threshold to binarize
		threshold(dframe, dframe, 30, 255, THRESH_BINARY);
		dilate(dframe, dframe, 0, Point(-1, -1), 2, 1, 1);
		erode(dframe, dframe, 0, Point(-1, -1), 2, 1, 1);
		dilate(dframe, dframe, 0, Point(-1, -1), 2, 1, 1);
		vector<vector<Point>> contours; // Vector for storing contour
		vector<Vec4i> hierarchy;

		findContours(dframe, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // Find the contours in the image
		// Display Image
		imshow("frame", dframe);

		int largest_area = 0;
		int largest_contour_index = 0;
		Rect bounding_rect;
		Scalar color(0, 0, 255);

		for (int i = 0; i < contours.size(); i++) // Iterate through each contour
		{
			double a = contourArea(contours[i], false); // Find the area of contour
			if (a > largest_area) {
				largest_area = a;
				largest_contour_index = i; // Store the index of largest contour
				bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
			}

		}
		for (int i = 0; i < contours.size(); i++) // Iterate through each contour
		{
			if (i != largest_contour_index) {
				drawContours(dframe, contours, i, color, FILLED, 8, hierarchy);
			}
		}

		Scalar color2(255, 0, 0);
		drawContours(dframe, contours, largest_contour_index, color2, 5, 8, hierarchy); // Draw the largest contour using previously stored index.
		rectangle(dframe, bounding_rect, Scalar(255, 255, 255), 2, 8, 0);
		imshow("Largest Contour", dframe);

		waitKey(20);
	}

	capture.release();
	return 0;
	/*
	Mat im1 = imread("moving.png");
	resize(im1, medianFrame, medianFrame.size(), 0, 0, INTER_CUBIC);
	imshow("moving image", im1);
	
	Mat diff = imread("moving.png");
	resize(diff, medianFrame, medianFrame.size(), 0, 0, INTER_CUBIC);
	absdiff(im1, medianFrame, diff);
	imshow("result image", diff);*/
	/*while (true) {
		
		bool ret = capture.read(frame);
		if (!ret) break;
		
		pMOG2->apply(frame, mask);
		pMOG2->getBackgroundImage(back_img);
		
		imshow("input", frame);
		imshow("mask", mask);
		imshow("background image", back_img);
		char c = waitKey(50);
		if (c == 27) {
			break;
		}
	}*/
	waitKey(0);
	return 0;
}
/*
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() == CV_8U);
	const int channels = I.channels();
	switch (channels)
	{
	case 1:
	{
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = table[I.at<uchar>(i, j)];
		break;
	}
	case 3:
	{
		Mat_<Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
			{
				_I(i, j)[0] = table[_I(i, j)[0]];
				_I(i, j)[1] = table[_I(i, j)[1]];
				_I(i, j)[2] = table[_I(i, j)[2]];
			}
		I = _I;
		break;
	}
	}

	return I;
}

Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() == CV_8U);
	int channels = I.channels();
	int nRows = I.rows;
	int nCols = I.cols * channels;
	if (I.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}
	int i, j;
	uchar* p;
	for (i = 0; i < nRows; ++i)
	{
		p = I.ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{
			p[j] = table[p[j]];
		}
	}
	return I;
}
void gammaCorrection(const Mat &img, const double gamma, int num)
{
	CV_Assert(gamma >= 0);
	//! [changing-contrast-brightness-gamma-correction]
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);

	Mat res = img.clone();
	LUT(img, lookUpTable, res);
	//! [changing-contrast-brightness-gamma-correction]
	Mat img_gamma_corrected;
	hconcat(img, res, img_gamma_corrected);
	string named = "Gamma correction" + num;
	imshow(named, img_gamma_corrected);
}
int computeMedian(vector<int> elements)
{
	nth_element(elements.begin(), elements.begin() + elements.size() / 2, elements.end());

	//sort(elements.begin(),elements.end());
	return elements[elements.size() / 2];
}

cv::Mat compute_median(std::vector<cv::Mat> vec)
{
	// Note: Expects the image to be CV_8UC3
	cv::Mat medianImg(vec[0].rows, vec[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int row = 0; row<vec[0].rows; row++)
	{
		for (int col = 0; col<vec[0].cols; col++)
		{
			std::vector<int> elements_B;
			std::vector<int> elements_G;
			std::vector<int> elements_R;

			for (int imgNumber = 0; imgNumber<vec.size(); imgNumber++)
			{
				int B = vec[imgNumber].at<cv::Vec3b>(row, col)[0];
				int G = vec[imgNumber].at<cv::Vec3b>(row, col)[1];
				int R = vec[imgNumber].at<cv::Vec3b>(row, col)[2];

				elements_B.push_back(B);
				elements_G.push_back(G);
				elements_R.push_back(R);
			}

			medianImg.at<cv::Vec3b>(row, col)[0] = computeMedian(elements_B);
			medianImg.at<cv::Vec3b>(row, col)[1] = computeMedian(elements_G);
			medianImg.at<cv::Vec3b>(row, col)[2] = computeMedian(elements_R);
		}
	}
	return medianImg;
}

int main(int argc, char const *argv[])
{
	std::string video_file;
	// Read video file
	video_file = "demo2.mp4";

	VideoCapture cap(video_file);
	if (!cap.isOpened())
		cerr << "Error opening video file\n";

	// Randomly select 25 frames
	default_random_engine generator;
	uniform_int_distribution<int>distribution(0,cap.get(CAP_PROP_FRAME_COUNT));

	vector<Mat> frames;
	Mat frame;

	for (int i = 0; i<25; i++)
	{
		int fid = distribution(generator);
		cap.set(CAP_PROP_POS_FRAMES, fid);
		Mat frame;
		cap >> frame;
		if (frame.empty())
			continue;
		frames.push_back(frame);
		cout << "keep" << i << endl;
	}
	// Calculate the median along the time axis
	cout << "caculating..." << endl;
	Mat medianFrame = compute_median(frames);
	cout << "result" << endl;
	// Display median frame
	imshow("frame", medianFrame);
	waitKey(0);
}*/