#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>

using namespace std;
using namespace cv;


Stitcher::Mode mode = Stitcher::PANORAMA;

Mat stitch_two_image(Mat original_image, Mat object_image);

int main()
{
	vector<Mat> frames;
	VideoCapture video("panorama_video_sampe2.mp4");

	int totalFrames = video.get(CAP_PROP_FRAME_COUNT);
	int skippingFrames = totalFrames / 20;

	if (skippingFrames == 0) skippingFrames = 1;

	cout << "Total frames:" << totalFrames << endl;

	for (int i = 0; i < totalFrames; i += skippingFrames) {
		Mat temp;
		video.set(CAP_PROP_POS_FRAMES, i);

		video >> temp;
		frames.push_back(temp);

		if (i % 10 == 0) cout << "Loading video :" << i << "/" << totalFrames << endl;
	}

	Mat lastFrame;
	video.set(CAP_PROP_POS_FRAMES, totalFrames - 1);
	video >> lastFrame;
	frames.push_back(lastFrame);

	for (int i = 0; i < frames.size(); i++) {
		resize(frames[i], frames[i], Size(0, 0), 0.5, 0.5, INTER_LINEAR);
	}

	Mat result = stitch_two_image(frames[0], frames[1]);

	for (int i = 2; i < frames.size(); i++) {
		result = stitch_two_image(result, frames[i]);
		cout << "Stitching image: " << i << "/" << frames.size() << endl;
	}

	imshow("result", result);

	waitKey(0);

	return 0;
}

Mat stitch_two_image(Mat original_image, Mat object_image) {
	// SIFT 특징점 검출기 초기화
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	// 특징점 및 기술자 추출
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	sift->detectAndCompute(original_image, cv::Mat(), keypoints1, descriptors1);
	sift->detectAndCompute(object_image, cv::Mat(), keypoints2, descriptors2);

	// BFMatcher로 특징점 매칭
	cv::BFMatcher bf(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	bf.match(descriptors1, descriptors2, matches);

	// 좋은 매칭 선택
	std::vector<cv::DMatch> good_matches;
	double min_dist = 100;
	double max_dist = 0;
	for (int i = 0; i < descriptors1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	for (int i = 0; i < descriptors1.rows; i++) {
		if (matches[i].distance < 2 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}

	// 좋은 매칭으로 객체 위치 찾기
	std::vector<cv::Point2f> src_pts, dst_pts;
	for (int i = 0; i < good_matches.size(); i++) {
		src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	// 변환 행렬 계산
	cv::Mat H = cv::findHomography(dst_pts, src_pts, cv::RANSAC);
	// 객체 이미지를 원본 이미지에 붙이기
	cv::Mat object_on_original;
	cv::warpPerspective(object_image, object_on_original, H, original_image.size());

	// 이미지 합성
	cv::Mat complementSet = original_image.clone(); // 원본 이미지 복제

	int max_pixel = complementSet.cols;
	bool max = true;


	// object_on_original를 뺀 부분을 검은색으로 채우기
	for (int i = 0; i < complementSet.rows; i++) {
		for (int j = 0; j < complementSet.cols; j++) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
				complementSet.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); // 검은색으로 설정
				if (max == true)
				{
					max_pixel = cv::min(max_pixel, j);
					max = false;
				}
			}
			max = true;
		}
	}

	//새로운 크기의 Mat 생성
	cv::Mat result = cv::Mat::zeros(object_image.rows, max_pixel + object_image.cols, CV_8UC3);

	//앞은 첫번째 이미지, 뒤는 두번째 이미지 채워넣기
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < max_pixel; j++) {
			result.at<cv::Vec3b>(i, j) = original_image.at<cv::Vec3b>(i, j);
		}
	}

	for (int i = 0; i < result.rows; i++) {
		for (int j = max_pixel; j < result.cols; j++) {
			result.at<cv::Vec3b>(i, j) = object_image.at<cv::Vec3b>(i, j - max_pixel);
		}
	}

	return result;
}