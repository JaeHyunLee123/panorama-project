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

	Mat result0 = stitch_two_image(frames[10], frames[11]);
	//Mat result1 = stitch_two_image(result0, frames[13]);

	//stitching�� ������ �� ��� �̹����� �������� ���������� ��ƼĪ�ϰ� flip�� ���� �������� ��ƼĪ �� ��ģ��.
	/*int midIndex = frames.size() / 2;
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

		for (int i = midIndex + 2; i < frames.size(); i++) {
		imshow("resultRight", resultRight);
		waitKey(1);
		resultRight = stitch_two_image(resultRight, frames[i]);
		cout << "Stitching image Right: " << i << "/" << frames.size() << endl;	
	}
	*/

	
	imshow("result0", result0);
	//imshow("result1", result1);
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

	int vsize = 0;
	if (matches.size() >= 50)
		vsize = 50;
	else
		vsize = matches.size();

	// ���� ��Ī ����
	std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + vsize);
	
		
	//// ���� ��Ī ����
	//std::vector<cv::DMatch> good_matches;
	//double min_dist = 50;
	//double max_dist = 0;

	//
	////min-max 
	//for (int i = 0; i < descriptors1.rows; i++) {
	//	double dist = matches[i].distance;
	//	if (dist < min_dist) min_dist = dist;
	//	if (dist > max_dist) max_dist = dist;
	//}

	//// good match ����
	//for (int i = 0; i < descriptors1.rows; i++) {
	//	if (matches[i].distance < 2 * min_dist) {
	//		good_matches.push_back(matches[i]);
	//	}
	//}
	
	

	// ���� ��Ī���� ��ü ��ġ ã��
	std::vector<cv::Point2f> src_pts, dst_pts;
	for (int i = 0; i < good_matches.size(); i++) {
		src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);		
	}

	
	//��Ī �ð�ȭ
	Mat visualMatching;
	drawMatches(originalCutImage, keypoints1, object_image, keypoints2, good_matches, visualMatching);
	imshow("matching point", visualMatching);


	// ��ȯ ��� ��� -> CV_64F
	Mat H = findHomography(dst_pts, src_pts, cv::RANSAC);

	// ��ȯ ����� �����ؼ� 0,100�� translate�� �����Ѵ�.
	double set[9] = {1,0, 0, 0, 1, 100, 0, 0, 1};
	Mat translate100 = Mat(H.size(), CV_64F, set);
	Mat translateH =  translate100 * H;

	// ��ȯ����� ������ object_on_original�� ���� 
	Mat object_on_original;
	cv::warpPerspective(object_image, object_on_original, translateH, Size(object_image.cols * 2, object_image.rows * 2), INTER_CUBIC);
	imshow("object_on_original", object_on_original);

	//originalCutImage�� object_image�� �ϳ��� ��ġ��
	//���� �������� ������ �����δ�.
	int topCol = 0;
	int bottomCol = 0;
	for (int i = 0; i < originalCutImage.rows; i++) {
		for (int j = 0; j < originalCutImage.cols; j++) {
			if(object_on_original.at<cv::Vec3b>(i + 100, j) ==  Vec3b(0,0,0))
				object_on_original.at<cv::Vec3b>(i + 100, j) = originalCutImage.at<cv::Vec3b>(i, j);
			else {
				for (int x = 0; x < 10; x++) {
					object_on_original.at<cv::Vec3b>(i + 100, j + x) = originalCutImage.at<cv::Vec3b>(i, j + x);					
				}
				
				//��輱 �� ���� ���� ���� ���� ���� ���� �����Ѵ�.
				if (i == 0)
					topCol = j;
				else if (i == originalCutImage.rows - 1)
					bottomCol = j;
				break;
			}
		}
	}
	imshow("object_on_original1", object_on_original);
	
	//���� �κ� �����
	//�ϴ� col�� ���� �۰��ϴ� �������� �غ�
	int minCol = object_on_original.cols;
	int minRow = 0;
	int maxCol = 0;
	int maxRow = 0;
	for (int i = object_on_original.rows - 1; i >= 0; i--) {
		for (int j = object_on_original.cols - 1; j >= 0; j--) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
				minCol = min(minCol, j);
				maxCol = max(maxCol, j);

				//minCol�̳� maxCol�� ���ŵǾ��� �� minRow�� maxRow �ʱ�ȭ
				if (minCol == j)
					minRow = i;
				if (maxCol == j)
					maxRow = i;
				break;
			}

		}
	}
	minCol++;
	maxCol++;

	//�̹��� ����
	//��ǥ ���� : ��ܿ��� ��, ��ܿ����� ��, �ϴܿ��� ��, �ϴܿ����� ��
	vector<cv::Point2f> inputPts, outputPts;
	if (minRow < maxRow) {
		inputPts.push_back(Point2f(100, topCol ));
		inputPts.push_back(Point2f(minRow, minCol));
		inputPts.push_back(Point2f(originalCutImage.rows - 1 + 100, bottomCol));
		inputPts.push_back(Point2f(maxRow , maxCol));
	}
	else {
		inputPts.push_back(Point2f(100, topCol));
		inputPts.push_back(Point2f(maxRow, maxCol));
		inputPts.push_back(Point2f(originalCutImage.rows - 1 + 100, bottomCol));
		inputPts.push_back(Point2f(minRow , minCol));
	}

	outputPts.push_back(Point2f(0, topCol));
	outputPts.push_back(Point2f(0, minCol));
	outputPts.push_back(Point2f(originalCutImage.rows - 1, bottomCol));
	outputPts.push_back(Point2f(originalCutImage.rows - 1, minCol));

	//��ȯ��� ����
	Mat M = getPerspectiveTransform(inputPts, outputPts);
	Mat anotherM;
	warpPerspective(object_on_original, anotherM, M, object_on_original.size(), INTER_CUBIC);
	imshow("M", anotherM);
	cout << endl;
	cout << "M : " << M << endl;

	//����� ������ mat ���� �� ������ �ű��
	//�߸� original�� object_on_original�� ���ľ� �Ѵ�.
	//�б⹮�� ���� ������ original_image.cols - originalCutImage.cols�� 0�̸� ������ �߻��ϱ� �����̴�.
	Mat result = Mat::zeros(object_on_original.rows, original_image.cols - originalCutImage.cols + minCol,CV_8UC3);
	if (original_image.cols - originalCutImage.cols == 0) {
		object_on_original(Rect(0, 0, minCol, originalCutImage.rows)).
			copyTo(result(Rect(original_image.cols - originalCutImage.cols, 0, minCol, originalCutImage.rows)));
	}
	else {
		original_image(Rect(0, 0, original_image.cols - originalCutImage.cols, originalCutImage.rows)).
			copyTo(result(Rect(0, 0, original_image.cols - originalCutImage.cols, originalCutImage.rows)));
		object_on_original(Rect(0, 0, minCol, originalCutImage.rows)).
			copyTo(result(Rect(original_image.cols - originalCutImage.cols, 0, minCol, originalCutImage.rows)));
	}
	waitKey(1);
	return result;
}
