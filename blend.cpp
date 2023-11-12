#include <opencv2/opencv.hpp>
#include <cmath>
#include "blend.h"

using namespace std;
using namespace cv;

namespace blend {

	Mat blendImage(Mat image, int center, int blendingArea) {
		Mat result = image.clone();

		for (int row = 0; row < image.rows; row++) {
			for (int i = 0; i < blendingArea; i++) {
				//center�� �������� ��Ī�Ǵ� �ȼ��� ���� �����´�
				int targetCol = center - blendingArea / 2 + i;
				int referenceCol = center + blendingArea / 2 - i;

				RGB targetRGB = getRGB(image, targetCol, row);
				RGB referenceRGB = getRGB(image, referenceCol, row);

				//������ �ȼ� ���� hsi ������ �ٲ۴�
				HSI targetHSI = rgb2hsi(targetRGB);
				HSI referenceHSI = rgb2hsi(referenceRGB);

				//intensity ������ ���� ������ �����Ѵ�.
				double alpha = (double)i / blendingArea;
				if (alpha > 0.5) {
					targetHSI.intensity = alpha * targetHSI.intensity + (1 - alpha) * referenceHSI.intensity;
				}
				else {
					targetHSI.intensity = (1 - alpha) * targetHSI.intensity + alpha * referenceHSI.intensity;
				}

				//�ٽ� rgb ������ �ٲ� �� ����
				targetRGB = hsi2rgb(targetHSI);

				putRGB(result, targetRGB, targetCol, row);
			}

			//�� �߰��� �ִ� �� ó��
			RGB centerRGB = getRGB(result, center, row);
			RGB leftRGB = getRGB(result, center - 1, row);
			RGB rightRGB = getRGB(result, center + 1, row);

			HSI centerHSI = rgb2hsi(centerRGB);
			HSI leftHSI = rgb2hsi(leftRGB);
			HSI rightHSI = rgb2hsi(rightRGB);

			centerHSI.intensity = (leftHSI.intensity + rightHSI.intensity) / 2;

			centerRGB = hsi2rgb(centerHSI);
			putRGB(result, centerRGB, center, row);
		}

		return result;
	}

	RGB getRGB(Mat image, int col, int row) {
		RGB result = { 0,0,0 };

		result.red = image.data[(col + row * image.cols) * 3 + 2];
		result.green = image.data[(col + row * image.cols) * 3 + 1];
		result.blue = image.data[(col + row * image.cols) * 3];

		return result;
	}

	void putRGB(Mat& dest, RGB rgb, int col, int row) {
		dest.data[(col + row * dest.cols) * 3 + 2] = rgb.red;
		dest.data[(col + row * dest.cols) * 3 + 1] = rgb.green;
		dest.data[(col + row * dest.cols) * 3] = rgb.blue;
	}

	RGB hsi2rgb(HSI input) {
		RGB result = { 0,0,0 };

		const double PI = 3.141592;

		//����� ���� ���� ����. �� ������ 0~1 ���� ���� ����. ���߿� 255 ���ؼ� ����
		double red = 0;
		double green = 0;
		double blue = 0;

		//����� ���� normalization
		double hue = input.hue * PI / 180.0;
		double saturation = input.saturation / 100.0;
		double intensity = input.intensity / 255.0;

		//���� ����
		if (hue < 2 * PI / 3) {
			blue = intensity * (1 - saturation);
			red = intensity * (1 + saturation * cos(hue) / cos(PI / 3 - hue));
			green = 3 * intensity - (red + blue);
		}
		else if (2 * PI / 3 <= hue && hue < 4 * PI / 3) {
			hue -= 2 * PI / 3;

			red = intensity * (1 - saturation);
			green = intensity * (1 + saturation * cos(hue) / cos(PI / 3 - hue));
			blue = 3 * intensity - (red + green);
		}
		else {
			hue -= 4 * PI / 3;

			green = intensity * (1 - saturation);
			blue = intensity * (1 + saturation * cos(hue) / cos(PI / 3 - hue));
			red = 3 * intensity - (blue + green);
		}

		result.red = red * 255;
		result.green = green * 255;
		result.blue = blue * 255;

		return result;
	}

	HSI rgb2hsi(RGB input) {
		HSI result = { 0,0,0 };
		const double PI = 3.141592;

		//����� ���� normalization
		double sum = input.red + input.green + input.blue;
		double red = input.red / sum;
		double green = input.green / sum;
		double blue = input.blue / sum;

		//����� ���� ���� ����. �� ������ 0~1 ���� ���� ����. ���߿� ���� 180/PI�� 100�� ���ؼ� ����
		double hue = 0;
		double saturation = 0;

		//���� ����
		double theta = acos(0.5 * ((red - green) + (red - blue)) / pow((pow(red - green, 2) + (red - blue) * (green - blue)), 0.5));
		if (blue > green) {
			hue = 2 * PI - theta;
		}
		else {
			hue = theta;
		}

		saturation = 1 - 3 * min(red, min(green, blue));

		result.hue = hue * 180 / PI;
		result.saturation = saturation * 100;
		result.intensity = (input.red + input.green + input.blue) / 3;

		return result;
	}
}