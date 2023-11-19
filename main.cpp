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

	Mat result0 = stitch_two_image(frames[11], frames[12]);

	/*for (int i = 13; i < 15; i++) {
		result0 = stitch_two_image(result0, frames[i]);
	}*/

	//Mat result1 = stitch_two_image(result0, frames[13]);

	//stitching�� ������ �� ��� �̹����� �������� ���������� ��ƼĪ�ϰ� flip�� ���� �������� ��ƼĪ ��  ��ģ��.
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

	// ORB Ư¡�� ����� �ʱ�ȭ
	int minHessian = 1000;
	cv::Ptr<cv::ORB> orb = cv::ORB::create(minHessian);

	// Ư¡�� �� ����� ����
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	orb->detectAndCompute(originalCutImage, noArray(), keypoints1, descriptors1);
	orb->detectAndCompute(object_image, noArray(), keypoints2, descriptors2);

	//-- Step 2: Matching descriptor vectors with a brute force matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	
	//���� ��Ī�� ����
	//1.Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.7f;
	std::vector<DMatch> first_good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i].size() == 2) {
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				first_good_matches.push_back(knn_matches[i][0]);
			}
		}	
	}


	//2. gradient�� ������
	// ��ü ��ġ ã��
	std::vector<cv::Point2f> src_pts, dst_pts;
	int gradientValue[11] = { 0, }; // -0.5~ 0.5�� 0.1��
	std::vector<std::vector<int>> gradientIndex(11); //�ش��ϴ� �ε��� ����
	float tmp = 0.0f;
	int tmpIndex = 0;

	for (int i = 0; i < first_good_matches.size(); i++) {
		src_pts.push_back(keypoints1[first_good_matches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[first_good_matches[i].trainIdx].pt);

		//gradient ��� �� ����
		tmp = ((float)(dst_pts[i].y - src_pts[i].y)) / ((float)(dst_pts[i].x - src_pts[i].x));
		tmpIndex = (int)(tmp * 10 + 5);
		if (tmpIndex >= 0 && tmpIndex < 11) {
			gradientValue[tmpIndex] = gradientValue[tmpIndex] + 1;
			gradientIndex[tmpIndex].push_back(i);
		}
			
	}

	//���� ū ���� ����� index�� �츮�� �������� ������.
	int maxIndex = 0;
	for (int i = 1; i < 11; i++) {
		if (gradientValue[maxIndex] < gradientValue[i])
			maxIndex = i;
	}

	std::vector<DMatch> second_good_matches;
	for (int i = 0; i < gradientValue[maxIndex]; i++) {
		second_good_matches.push_back(first_good_matches[gradientIndex[maxIndex][i]]);
	}

	//3. �ִ� �Ÿ��� ���� ���� ����(������ �Ÿ�)
	std::vector<cv::DMatch> third_good_matches;
	std::vector<double> pointDistance;
	double distanceMean = 0;
	double distanceTmp = 0;
	Point2f oriPoint, objPoint;
	for (int i = 0; i < second_good_matches.size(); i++) {
		oriPoint = keypoints1[second_good_matches[i].queryIdx].pt;
		objPoint = keypoints2[second_good_matches[i].trainIdx].pt;

		//�Ÿ� ���� ���
		distanceTmp = sqrt(pow(oriPoint.x - objPoint.x - originalCutImage.cols, 2) + pow(oriPoint.y - objPoint.y, 2));
		pointDistance.push_back(distanceTmp);
		distanceMean += distanceTmp;
		cout << distanceTmp << endl;
	}

	//����� +-20�ۼ�Ʈ ������ ����� ����
	distanceMean /= second_good_matches.size();

	for (int i = 0; i < second_good_matches.size(); i++) {
		oriPoint = keypoints1[second_good_matches[i].queryIdx].pt;
		objPoint = keypoints2[second_good_matches[i].trainIdx].pt;

		//�Ÿ� ���� ���
		distanceTmp = sqrt(pow(oriPoint.x - objPoint.x - originalCutImage.cols, 2) + pow(oriPoint.y - objPoint.y, 2));
		if (distanceTmp < distanceMean * 1.1 && distanceTmp > distanceMean * 0.9)
			third_good_matches.push_back(second_good_matches[i]);
	}


	// ���� ��Ī���� ��ü ��ġ ã��
	src_pts.clear();
	dst_pts.clear();
	for (int i = 0; i < second_good_matches.size(); i++) {
		src_pts.push_back(keypoints1[second_good_matches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[second_good_matches[i].trainIdx].pt);
	}


	//��Ī�� �ð�ȭ
	Mat img_matches;
	drawMatches(originalCutImage, keypoints1, object_image, keypoints2, first_good_matches, img_matches);
	Mat img_matches2;
	drawMatches(originalCutImage, keypoints1, object_image, keypoints2, second_good_matches, img_matches2);
	Mat img_matches3;
	drawMatches(originalCutImage, keypoints1, object_image, keypoints2, third_good_matches, img_matches3);
	imshow("img_matches", img_matches);
	imshow("img_matches2", img_matches2);
	imshow("img_matches3", img_matches3);


	// ��ȯ ��� ��� -> CV_64F
	Mat H = findHomography(dst_pts, src_pts, cv::RANSAC);

	// ��ȯ ����� �����ؼ� 0,100�� translate�� �����Ѵ�.
	double set[9] = {1,0, 0, 0, 1, 100, 0, 0, 1};
	Mat translate100 = Mat(H.size(), CV_64F, set);
	Mat translateH =  translate100 * H;

	// ��ȯ����� ������ object_on_original�� ���� 
	Mat object_on_original = Mat::zeros(object_image.cols * 2, object_image.rows * 2, CV_8UC3);
	cv::warpPerspective(object_image, object_on_original, translateH, Size(object_image.cols * 2, object_image.rows * 2), INTER_CUBIC);
	imshow("object_on_original", object_on_original);

	//originalCutImage�� object_image�� �ϳ��� ��ġ��
	//���� �������� ������ �����δ�.
	//�̶� ���� ���� row�� ū row���� ������ ��ǥ�� �����Ѵ�.
	int topCol = 0;
	int bottomCol = 0;
	for (int i = 0; i < originalCutImage.rows; i++) {
		for (int j = 0; j < originalCutImage.cols; j++) {
			if(object_on_original.at<cv::Vec3b>(i + 100, j) ==  Vec3b(0,0,0))
				object_on_original.at<cv::Vec3b>(i + 100, j) = originalCutImage.at<cv::Vec3b>(i, j);
			else {
				for (int x = 0; x < 15; x++) {
					//�ε��� ������ ���� ���� ���ǹ�
					if(j + x < originalCutImage.cols)
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

	
	//�̹����� ������ �κп��� Ƣ��� �κ��� �����ϱ� ���� �ݺ���
	//�� ��ǥ�� ���� ���� 4���� ����
	int minCol = object_on_original.cols - 1;
	int minRow = 0;
	int maxCol = 0;
	int maxRow = 0;

	//������ �κ��� gradient�� ���ϱ� ���� ����
	bool isPlusGradient;

	//gradient�� ���ϱ� ���� ����
	//����� �ݺ����� 2�� ����ؾ� ������ ���Ŀ� ������ �Ǹ� ������ ����
	for (int i = object_on_original.rows - 1; i >= 0; i--) {
		for (int j = object_on_original.cols - 1; j >= 0; j--) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {			
				//minCol�̳� maxCol�� ���ŵǾ��� �� minRow�� maxRow �ʱ�ȭ
				//minCol�� j�� ������ �ְ� �̶� �� ������ ���´�. ������ minMax ����� ������
				if (minCol > j) {
					minCol = j;
					minRow = i;
				}

				if (maxCol < j) {
					maxCol = j;
					maxRow = i;
				}
				break;
			}
		}
	}

	//�� ��ǥ�� ���ؼ� gradient�� ���Ѵ�.
	//�� ��ǥ�� �������� Row�� �����ϰ� Col�� �Ȱ��� -> ���� �з��� �ʿ� ����
	if (minRow >= maxRow)
		isPlusGradient = true;
	else
		isPlusGradient = false;


	//gradient�� ������� ��ǥ ����
	//isPlusGradient�� true�� ��쿡�� max�� ��� �����ϰ� min�� ���� ó�� ��ǥ�� ����
	//false�� ��쿡�� �ݴ��
	for (int i = object_on_original.rows - 1; i >= 0; i--) {
		for (int j = object_on_original.cols - 1; j >= 0; j--) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
				//minCol�̳� maxCol�� ���ŵǾ��� �� minRow�� maxRow �ʱ�ȭ
				if (isPlusGradient == true) {
					if (minCol > j) {
						minCol = j;
						minRow = i;
					}

					if (maxCol <= j) {
						maxCol = j;
						maxRow = i;
					}
				}
				else {
					if (minCol >= j) {
						minCol = j;
						minRow = i;
					}

					if (maxCol < j) {
						maxCol = j;
						maxRow = i;
					}
				}				
				break;
			}
		}
	}

	//�̹��� ����
	//��ǥ ���� : ��ܿ��� ��, ��ܿ����� ��, �ϴܿ��� ��, �ϴܿ����� ��
	//��Ŀ��� ������ (row, col)������ ��ǥ�� (x,y)�̰� x�� ���� y�� ���� �����̴�.
	vector<cv::Point2f> inputPts, outputPts;
	if (minRow < maxRow) {
		inputPts.push_back(Point2f(topCol, 100));
		inputPts.push_back(Point2f(minCol, minRow));
		inputPts.push_back(Point2f(bottomCol, originalCutImage.rows - 1 + 100));
		inputPts.push_back(Point2f(maxCol, maxRow));
	}
	else {
		inputPts.push_back(Point2f(topCol, 100));
		inputPts.push_back(Point2f(maxCol, maxRow));
		inputPts.push_back(Point2f(bottomCol, originalCutImage.rows - 1 + 100));
		inputPts.push_back(Point2f(minCol, minRow));
	}

	////�� ����
	//circle(object_on_original, Point2f(topCol, 100), 3, Scalar(0, 0, 255), 3);
	//circle(object_on_original, Point2f(minCol, minRow), 3, Scalar(0, 255, 0), 3);
	//circle(object_on_original, Point2f(bottomCol, originalCutImage.rows - 1 + 100), 3, Scalar(255, 0, 0), 3);
	//circle(object_on_original, Point2f(maxCol, maxRow), 3, Scalar(255, 255, 255), 3);
	

	outputPts.push_back(Point2f(topCol, 100));
	outputPts.push_back(Point2f(minCol, 100));
	outputPts.push_back(Point2f(bottomCol, originalCutImage.rows - 1 + 100));
	outputPts.push_back(Point2f(minCol, originalCutImage.rows - 1 + 100));
	imshow("object_on_original2", object_on_original);


	//��ȯ��� ����
	Mat M = getPerspectiveTransform(inputPts, outputPts);
	Mat anotherM;
	warpPerspective(object_on_original, anotherM, M, object_on_original.size(), INTER_CUBIC);
	imshow("M", anotherM);

	//M�� ������ �˾ƾߵȴ�.
	int mRightCols =0;
	for (int i = anotherM.rows - 1; i >= 0; i--) {
		for (int j = anotherM.cols -1; j >= 0; j--) {
			if (anotherM.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
				mRightCols = max(mRightCols, j);
		}
	}

	////object_on_original�� M�� ������ Mat ���� �� �� ����
	////�������� �Ķ����� ���� ������ ���� �� �̹����� ��ģ��.
	//float line = (float)(topCol - bottomCol) / (float)originalCutImage.rows;
	//Mat combineImg(originalCutImage.rows, minCol, CV_8UC3);
	//for (int i = 0; i < originalCutImage.rows; i++) {
	//	for (int j = 0; j < minCol; j++) {
	//		//topCol�� bottomCol���� ū ��� ���ְ� �ݴ��� ��쿡�� ���Ѵ�.
	//		if (topCol >= bottomCol) {
	//			if (i < topCol - (int)(line * i))
	//				combineImg.at<Vec3b>(i,j) = object_on_original.at<Vec3b>(i + 100, j);
	//			else
	//				combineImg.at<Vec3b>(i, j) = anotherM.at<Vec3b>(i + 100, j);
	//		}
	//		else {
	//			if (i < topCol + (int)(line * i))
	//				combineImg.at<Vec3b>(i, j) = object_on_original.at<Vec3b>(i + 100, j);
	//			else
	//				combineImg.at<Vec3b>(i, j) = anotherM.at<Vec3b>(i + 100, j);
	//		}		
	//	}
	//}
	//object_on_original�� M�� ������ Mat ���� �� �� ����
	//topCol�� bottomCol �� ���ݸ�
	//ó������ object�� topCol�� bottomCol �� ���� ����ŭ ���δ�.
	//���� topCol�� bottomCol �� ���ݸ�ŭ object�� ���̰� �������� M�� mRightCols���� ���δ�.
	Mat combineImg(originalCutImage.rows, mRightCols, CV_8UC3);
	int abCol = abs(topCol - bottomCol)/2;
	int minTopBottomCol = min(topCol, bottomCol);

	for (int i = 0; i < originalCutImage.rows; i++) {
		for (int j = 0; j < minTopBottomCol + abCol; j++) {
			combineImg.at<Vec3b>(i, j) = object_on_original.at<Vec3b>(i + 100, j);
		}
	}
	for (int i = 0; i < originalCutImage.rows; i++) {
		for (int j = minTopBottomCol + abCol; j < mRightCols; j++) {
			combineImg.at<Vec3b>(i, j) = anotherM.at<Vec3b>(i + 100, j);
		}
	}

	imshow("combine", combineImg);

	//����� ������ mat ���� �� ������ �ű��
	//�߸� original�� object_on_original�� ���ľ� �Ѵ�.
	//�б⹮�� ���� ������ original_image.cols - originalCutImage.cols�� 0�̸� ������ �߻��ϱ� �����̴�.
	Mat result = Mat::zeros(combineImg.rows, original_image.cols - originalCutImage.cols + combineImg.cols,CV_8UC3);
	if (original_image.cols - originalCutImage.cols == 0) {
		combineImg(Rect(0, 0, minCol, originalCutImage.rows)).
			copyTo(result(Rect(original_image.cols - originalCutImage.cols, 0, minCol, originalCutImage.rows)));
	}
	else {
		original_image(Rect(0, 0, original_image.cols - originalCutImage.cols, originalCutImage.rows)).
			copyTo(result(Rect(0, 0, original_image.cols - originalCutImage.cols, originalCutImage.rows)));
		combineImg(Rect(0, 0, minCol, originalCutImage.rows)).
			copyTo(result(Rect(original_image.cols - originalCutImage.cols, 0, minCol, originalCutImage.rows)));
	}
	waitKey(1);
	return result;
}
