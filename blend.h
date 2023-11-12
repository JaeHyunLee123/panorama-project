#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

namespace blend{
	struct RGB {
		int red;
		int green;
		int blue;
	};

	struct HSI
	{
		int hue; // 0 ~ 180/PI 값을 가짐
		int saturation; //0 ~ 100
		int intensity; // 0 ~ 255
	};

	//mergedImage: blending 할 대상 이미지
	//center: 접해진 부분
	//blendingArea: blending 할 x축 길이
	//errorRange: center 추정 위치의 오차 범위
	Mat blendImage(Mat mergedImage, int center, int blendingArea, int errorRange);

	RGB getRGB(Mat image, int col, int row);
	void putRGB(Mat& dest, RGB rgb, int col, int row);

	RGB hsi2rgb(HSI input);
	HSI rgb2hsi(RGB input);
}