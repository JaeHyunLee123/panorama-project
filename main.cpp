#include <opencv2/opencv.hpp>
#include <iostream>
#include <array>
#include <cmath>
#include "blend.h"

using namespace std;
using namespace cv;



int main()
{
	Mat sample = imread("sample.png");

	int blendingArea = 200;
	int errorRange = 20;

	Mat blend = blend::blendImage(sample, sample.cols / 2, blendingArea, errorRange);

	//blending 영역 표시
	line(blend, Point(sample.cols / 2 - blendingArea/2, sample.rows / 2), Point(sample.cols / 2 + blendingArea / 2, sample.rows / 2), Scalar(0, 0, 0));
	//오차 범위 영역 표시
	line(blend, Point(sample.cols / 2 - errorRange, sample.rows / 2 + 20), Point(sample.cols / 2 + errorRange, sample.rows / 2 + 20), Scalar(0, 0, 0));

	imshow("input", sample);
	imshow("blend", blend);

	waitKey(0);

	return 0;
}


