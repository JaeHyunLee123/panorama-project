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

	Mat blend = blend::blendImage(sample, sample.cols / 2, 200);

	imshow("blend", blend);
	imwrite("blend.png", blend);

	waitKey(0);

	return 0;
}


