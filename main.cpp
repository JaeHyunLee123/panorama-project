#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>

int index12 = 1;

using namespace std;
using namespace cv;

Mat stitch_two_image(Mat original_image, Mat object_image);

int main()
{
	vector<Mat> frames;
	VideoCapture video("panorama_video_sampe2.mp4");

	int totalFrames = video.get(CAP_PROP_FRAME_COUNT);
	float skippingFrames = totalFrames / 19.0;

	if (skippingFrames == 0) skippingFrames = 1;

	cout << "Total frames:" << totalFrames << endl;

	for (float i = 0; i < totalFrames; i += skippingFrames) {
		Mat temp;
		video.set(CAP_PROP_POS_FRAMES, static_cast<int>(i));

		video >> temp;
		frames.push_back(temp);

		if (static_cast<int>(i) % 10 == 0) cout << "Loading video :" << static_cast<int>(i) << "/" << totalFrames << endl;
	}
	

	for (int i = 0; i < frames.size(); i++) {
		resize(frames[i], frames[i], Size(0, 0), 0.5, 0.5, INTER_LINEAR);
	}


	/*
	midIndex += 3;
	Mat result1 = stitch_two_image(frames[midIndex], frames[midIndex + 1]);
	flip(result1, result1, 1);
	flip(frames[midIndex - 1], frames[midIndex - 1], 1);
	result1 = stitch_two_image(result1, frames[midIndex - 1]);
	flip(result1, result1, 1);
	cout << "Stitching image Right: " << endl;

	midIndex += 3;
	Mat result2 = stitch_two_image(frames[midIndex], frames[midIndex + 1]);
	flip(result2, result2, 1);
	flip(frames[midIndex - 1], frames[midIndex - 1], 1);
	result2 = stitch_two_image(result2, frames[midIndex - 1]);
	flip(result2, result2, 1);
	cout << "Stitching image Right: " << endl;

	Mat result3 = stitch_two_image(result1, result2);
	flip(result3, result3, 1);
	flip(result0, result0, 1);
	result3 = stitch_two_image(result3, result0);
	flip(result3, result3, 1);
	cout << "Stitching image Right: " << endl;
	
	*/
	//stitching�� ������ �� ��� �̹����� �������� ���������� ��ƼĪ�ϰ� flip�� ���� �������� ��ƼĪ �� ��ģ��.
	int midIndex = frames.size() / 2;
	Mat result0 = stitch_two_image(frames[midIndex], frames[midIndex + 1]);
	flip(result0, result0, 1);
	flip(frames[midIndex - 1], frames[midIndex - 1], 1);
	result0 = stitch_two_image(result0, frames[midIndex - 1]);
	flip(result0, result0, 1);
	cout << "Stitching image Right: "  << endl;

	
	
	//������ ���� ����
	//Mat resultRight = stitch_two_image(frames[midIndex], frames[midIndex + 1]);
	//Mat resultRight1 = stitch_two_image(resultRight, frames[midIndex + 2]);

	//���� ���� ����
	//Mat resultRight2 = stitch_two_image(resultRight1, frames[midIndex + 3]);
	//Mat resultRight3 = stitch_two_image(resultRight2, frames[midIndex + 4]);
	/*
		for (int i = midIndex + 2; i < frames.size(); i++) {
		imshow("resultRight", resultRight);
		waitKey(1);
		resultRight = stitch_two_image(resultRight, frames[i]);
		cout << "Stitching image Right: " << i << "/" << frames.size() << endl;	
	}
	*/
	
	
	
	//imshow("result3", result3);
	//imwrite("9_17.jpg", result3);
	waitKey(0);

	return 0;
}



Mat stitch_two_image(Mat original_image, Mat object_image) {
	//��ü ��Ī �� �� �̹����� ũ�⸦ �Ȱ��� �ؼ� �߸��� ��Ī�� ���� ���̷��� ��
	Mat originalCutImage(object_image.size(), CV_8UC3);
	original_image(Rect(original_image.cols - originalCutImage.cols, 0, originalCutImage.cols, originalCutImage.rows)).
		copyTo(originalCutImage(Rect(0, 0, originalCutImage.cols, originalCutImage.rows)));

	// SIFT Ư¡�� ����� �ʱ�ȭ
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	// Ư¡�� �� ����� ����
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	sift->detectAndCompute(originalCutImage, cv::Mat(), keypoints1, descriptors1);
	sift->detectAndCompute(object_image, cv::Mat(), keypoints2, descriptors2);

	// BFMatcher�� Ư¡�� ��Ī
	cv::BFMatcher bf(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	bf.match(descriptors1, descriptors2, matches);

	
	std::sort(matches.begin(), matches.end());

	int vSize = 0;
	if (matches.size() >= 50)
		vSize = 50;
	else
		vSize = matches.size();

	// ���� ��Ī ����
	std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + vSize);
	
	

	/*
	// ���� ��Ī ����
	std::vector<cv::DMatch> good_matches;
	double min_dist = 30;
	double max_dist = 0;

	
	//min-max 
	for (int i = 0; i < descriptors1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	// good match ����
	for (int i = 0; i < descriptors1.rows; i++) {
		if (matches[i].distance < 2 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}
	*/
	



	// ���� ��Ī���� ��ü ��ġ ã��
	std::vector<cv::Point2f> src_pts, dst_pts;
	for (int i = 0; i < good_matches.size(); i++) {
		src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);		
	}

	
	//��Ī �ð�ȭ
	Mat visualMatching;
	drawMatches(originalCutImage, keypoints1, object_image, keypoints2, good_matches, visualMatching);
	//imshow("matching point", visualMatching);
	string str = "Result" + to_string(index12) + ".jpg";
	imwrite(str, visualMatching);

	// ��ȯ ��� ���
	Mat H = findHomography(dst_pts, src_pts, cv::RANSAC);

	// ��ȯ����� ������ object_on_original�� ���� 
	cv::Mat object_on_original;
	cv::warpPerspective(object_image, object_on_original, H, Size(object_image.cols * 2, object_image.rows), INTER_CUBIC);
	
	
	//imshow("object_on_original", object_on_original);
	
	int max_pixel = 0;
	// object_on_original���� ���� �κ� �� ���� �� �κ��� ã�� ���� �ݺ���
	//���Ʒ��� �ָ��ϰ� ���� �κ��� ����� ����
	for (int i = 0; i < object_on_original.rows; i++) {
		for (int j = 0; j < object_on_original.cols; j++) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
				max_pixel = cv::max(max_pixel, j);
				break;
			}
		}
	}
	
	
	//originalCutImage�� object_image�� �ϳ��� ��ġ��
	for (int i = 0; i < originalCutImage.rows; i++) {
		for (int j = 0; j < originalCutImage.cols; j++) {
			object_on_original.at<cv::Vec3b>(i, j) = originalCutImage.at<cv::Vec3b>(i, j);
		}
	}

	
	//���� �κ� �����
	//�ϴ� col�� ���� �۰��ϴ� �������� �غ�
	int min_pixel = object_on_original.cols;
	for (int i = object_on_original.rows - 1; i >= 0; i--) {
		for (int j = object_on_original.cols - 1; j >= 0; j--) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
				min_pixel = cv::min(min_pixel, j);
				break;
			}

		}
	}
	min_pixel++;

	//����� ������ mat ���� �� ������ �ű��
	//�߸� original�� object_on_original�� ���ľ� �Ѵ�.
	//�б⹮�� ���� ������ original_image.cols - originalCutImage.cols�� 0�̸� ������ �߻��ϱ� �����̴�.
	Mat result = Mat::zeros(object_on_original.rows, original_image.cols - originalCutImage.cols + min_pixel,CV_8UC3);
	if (original_image.cols - originalCutImage.cols == 0) {
		object_on_original(Rect(0, 0, min_pixel, originalCutImage.rows)).
			copyTo(result(Rect(original_image.cols - originalCutImage.cols, 0, min_pixel, originalCutImage.rows)));
	}
	else {
		original_image(Rect(0, 0, original_image.cols - originalCutImage.cols, originalCutImage.rows)).
			copyTo(result(Rect(0, 0, original_image.cols - originalCutImage.cols, originalCutImage.rows)));
		object_on_original(Rect(0, 0, min_pixel, originalCutImage.rows)).
			copyTo(result(Rect(original_image.cols - originalCutImage.cols, 0, min_pixel, originalCutImage.rows)));
	}
	waitKey(1);
	return result;
}
