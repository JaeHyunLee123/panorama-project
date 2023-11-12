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
		int hue;
		int saturation;
		int intensity;
	};

	Mat blendImage(Mat mergedImage, int center, int _blendingArea);

	RGB getRGB(Mat image, int col, int row);
	void putRGB(Mat& dest, RGB rgb, int col, int row);

	RGB hsi2rgb(HSI input);
	HSI rgb2hsi(RGB input);
}