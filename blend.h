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
		int hue; // 0 ~ 180/PI ���� ����
		int saturation; //0 ~ 100
		int intensity; // 0 ~ 255
	};

	//mergedImage: blending �� ��� �̹���
	//center: ������ �κ�
	//blendingArea: blending �� x�� ����
	//errorRange: center ���� ��ġ�� ���� ����
	Mat blendImage(Mat mergedImage, int center, int blendingArea, int errorRange);

	RGB getRGB(Mat image, int col, int row);
	void putRGB(Mat& dest, RGB rgb, int col, int row);

	RGB hsi2rgb(HSI input);
	HSI rgb2hsi(RGB input);
}