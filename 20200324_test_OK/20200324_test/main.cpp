#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <windows.h>
#include <math.h>
using namespace std;
using namespace cv;
double alpha; /**< Simple contrast control */
int beta;  /**< Simple brightness control */



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
	string named = "Gamma correction"+num;
	imshow(named, img_gamma_corrected);
}

int main()
{
	int divideWith = 255; // convert our input string to number - C++ style

	double t = (double)getTickCount();
	Mat image = imread("building.jpg");
	imshow("image", image);

	Mat image2 = imread("unnamed.jpg");
	imshow("image2", image2);

	gammaCorrection(image, 0.2, 1);
	gammaCorrection(image2, 3.0, 2);
	/*uchar table[256];
	for (int i = 0; i < 256; ++i)
		table[i] = (uchar)(pow(i / 255.0, 0.2) * 255.0);;//(divideWith * (i / divideWith));

	Mat outimg = ScanImageAndReduceRandomAccess(image, table);
	imshow("outimg", outimg);


	for (int i = 0; i < 256; ++i)
		table[i] = (uchar)(pow(i / 255.0, 3) * 255.0);;//(divideWith * (i / divideWith));


	Mat outimg2 = ScanImageAndReduceRandomAccess(image2, table);
	imshow("outimg2", outimg2);*/

	// do something ...
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Times passed in seconds: " << t << endl;

	waitKey();
	return 0;
	/*
	/// Read image given by user
	Mat image = imread("moon.jpg");
	Mat new_image = Mat::zeros(image.size(), image.type());

	/// Initialize values
	std::cout << " Basic Linear Transforms " << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "* Enter the alpha value [1.0-3.0]: "; std::cin >> alpha;
	std::cout << "* Enter the beta value [0-100]: "; std::cin >> beta;

	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for (int y = 0; y < image.rows; y++)
	{
	for (int x = 0; x < image.cols; x++)
	{
	for (int c = 0; c < 3; c++)
	{
	new_image.at<Vec3b>(y, x)[c] =
	saturate_cast<uchar>(alpha*(image.at<Vec3b>(y, x)[c]) + beta);
	}
	}
	}

	/// Create Windows
	namedWindow("Original Image", 1);
	namedWindow("New Image", 1);

	/// Show stuff
	imshow("Original Image", image);
	imshow("New Image", new_image);

	/// Wait until user press some key
	waitKey();
	return 0;*/
}