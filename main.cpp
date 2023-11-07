#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>


using namespace std;
using namespace cv;

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

	Mat result = stitch_two_image(frames[10], frames[11]);
	//Mat result2 = stitch_two_image(result, frames[9]);
	//Mat result3 = stitch_two_image(result2, frames[10]);

	/*
	for (int i = 2; i < 9; i++) {
		result = stitch_two_image(result, frames[i]);
		cout << "Stitching image: " << i << "/" << frames.size() << endl;
	}
	*/
	
	

	
	
	
	imshow("result", result);
	//imshow("result2", result2);
	//imshow("result3", result3);
	//imshow("0", frames[0]);
	//imshow("1", frames[1]);
	//imshow("2", frames[2]);
	//imshow("3", frames[3]);
	//imshow("4", frames[4]);
	//imshow("5", frames[5]);
	//imshow("6", frames[6]);
	imshow("7", frames[7]);
	imshow("8", frames[8]);
	imshow("9", frames[9]);
	imshow("10", frames[10]);

	waitKey(0);

	return 0;
}



Mat stitch_two_image(Mat original_image, Mat object_image) {
	//객체 매칭 시 두 이미지의 크기를 똑같이 해서 잘못된 매칭의 수를 줄이려고 함
	Mat originalCutImage(object_image.size(), CV_8UC3);
	original_image(Rect(original_image.cols - originalCutImage.cols, 0, originalCutImage.cols, originalCutImage.rows)).
		copyTo(originalCutImage(Rect(0, 0, originalCutImage.cols, originalCutImage.rows)));

	// SIFT 특징점 검출기 초기화
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	// 특징점 및 기술자 추출
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	sift->detectAndCompute(originalCutImage, cv::Mat(), keypoints1, descriptors1);
	sift->detectAndCompute(object_image, cv::Mat(), keypoints2, descriptors2);

	// BFMatcher로 특징점 매칭
	cv::BFMatcher bf(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	bf.match(descriptors1, descriptors2, matches);


	std::sort(matches.begin(), matches.end());

	int vSize = 0;
	if (matches.size() >= 50)
		vSize = 50;
	else
		vSize = matches.size();

			// 좋은 매칭 선택
	std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + vSize);
	
	/*
	// 좋은 매칭 선택
	std::vector<cv::DMatch> good_matches;
	double min_dist = 30;
	double max_dist = 0;

	
	//min-max 
	for (int i = 0; i < descriptors1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	// good match 실행
	for (int i = 0; i < descriptors1.rows; i++) {
		if (matches[i].distance < 2 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}
*/


	// 좋은 매칭으로 객체 위치 찾기
	std::vector<cv::Point2f> src_pts, dst_pts;
	for (int i = 0; i < good_matches.size(); i++) {
		src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);		
	}

	
	//매칭 시각화
	Mat visualMatching;
	drawMatches(original_image, keypoints1, object_image, keypoints2, good_matches, visualMatching);
	imshow("matching point", visualMatching);
	



	// 변환 행렬 계산
	Mat H = findHomography(dst_pts, src_pts, cv::RANSAC);

	// 변환행렬을 적용해 object_on_original에 저장 
	cv::Mat object_on_original;
	cv::warpPerspective(object_image, object_on_original, H, Size(object_image.cols * 2, object_image.rows), INTER_CUBIC);
	
	
	imshow("object_on_original", object_on_original);
	/*
	int max_pixel = 0;
	// object_on_original에서 검정 부분 중 가장 긴 부분을 찾기 위한 반복문
	//위아래의 애매하게 검은 부분을 지우기 위함
	for (int i = 0; i < object_on_original.rows; i++) {
		for (int j = 0; j < object_on_original.cols; j++) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
				max_pixel = cv::max(max_pixel, j);
				break;
			}
		}
	}
	*/
	
	
	//originalCutImage와 object_image를 하나로 합치기
	for (int i = 0; i < originalCutImage.rows; i++) {
		for (int j = 0; j < originalCutImage.cols; j++) {
			object_on_original.at<cv::Vec3b>(i, j) = originalCutImage.at<cv::Vec3b>(i, j);
		}
	}

	
	//검은 부분 지우기
	//일단 col을 가장 작게하는 방향으로 해봄
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

	//결과를 저장할 mat 생성 후 데이터 옮기기
	//잘린 original과 object_on_original를 합쳐야 한다.
	//분기문을 넣은 이유는 original_image.cols - originalCutImage.cols가 0이면 오류가 발생하기 때문이다.
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
