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
	Mat result1 = stitch_two_image(result0, frames[13]);

	//stitching을 수행할 때 가운데 이미지를 기준으로 오른쪽으로 스티칭하고 flip을 통해 왼쪽으로 스티칭 후  합친다.
	/*int midIndex = frames.size() / 2;
	Mat result0 = stitch_two_image(frames[midIndex], frames[midIndex + 1]);
	flip(result0, result0, 1);
	flip(frames[midIndex - 1], frames[midIndex - 1], 1);
	result0 = stitch_two_image(result0, frames[midIndex - 1]);
	flip(result0, result0, 1);
	cout << "Stitching image Right: "  << endl;
	
	//오른쪽 연산 수행
	//Mat resultRight = stitch_two_image(frames[midIndex], frames[midIndex + 1]);
	//Mat resultRight1 = stitch_two_image(resultRight, frames[midIndex + 2]);

	//왼쪽 연산 수행
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
	imshow("result1", result1);
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

	
	/*std::sort(matches.begin(), matches.end());

	int vsize = 0;
	if (matches.size() >= 50)
		vsize = 50;
	else
		vsize = matches.size();*/

	// 좋은 매칭 선택
	//std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + vsize);
	
		
	// 좋은 매칭 선택
	std::vector<cv::DMatch> good_matches;
	double min_dist = 50;
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
	
	

	// 좋은 매칭으로 객체 위치 찾기
	std::vector<cv::Point2f> src_pts, dst_pts;
	for (int i = 0; i < good_matches.size(); i++) {
		src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);		
	}

	
	//매칭 시각화
	Mat visualMatching;
	drawMatches(originalCutImage, keypoints1, object_image, keypoints2, good_matches, visualMatching);
	imshow("matching point", visualMatching);


	// 변환 행렬 계산 -> CV_64F
	Mat H = findHomography(dst_pts, src_pts, cv::RANSAC);

	// 변환 행렬을 수정해서 0,100의 translate를 적용한다.
	double set[9] = {1,0, 0, 0, 1, 100, 0, 0, 1};
	Mat translate100 = Mat(H.size(), CV_64F, set);
	Mat translateH =  translate100 * H;

	// 변환행렬을 적용해 object_on_original에 저장 
	Mat object_on_original;
	cv::warpPerspective(object_image, object_on_original, translateH, Size(object_image.cols * 2, object_image.rows * 2), INTER_CUBIC);
	imshow("object_on_original", object_on_original);

	//originalCutImage와 object_image를 하나로 합치기
	//검은 영역에만 영상을 덧붙인다.
	//이때 가장 작은 row와 큰 row에서 만나는 좌표를 저장한다.
	int topCol = 0;
	int bottomCol = 0;
	for (int i = 0; i < originalCutImage.rows; i++) {
		for (int j = 0; j < originalCutImage.cols; j++) {
			if(object_on_original.at<cv::Vec3b>(i + 100, j) ==  Vec3b(0,0,0))
				object_on_original.at<cv::Vec3b>(i + 100, j) = originalCutImage.at<cv::Vec3b>(i, j);
			else {
				for (int x = 0; x < 10; x++) {
					//인덱스 오버를 막기 위한 조건문
					if(j + x < originalCutImage.cols)
						object_on_original.at<cv::Vec3b>(i + 100, j + x) = originalCutImage.at<cv::Vec3b>(i, j + x);					
				}
				
				//경계선 중 가장 위의 값과 가장 밑의 값을 저장한다.
				if (i == 0)
					topCol = j;
				else if (i == originalCutImage.rows - 1)
					bottomCol = j;
				break;
			}
		}
	}
	imshow("object_on_original1", object_on_original);
	
	//이미지의 오른쪽 부분에서 튀어나온 부분을 검출하기 위한 반복문
	//두 좌표의 값을 위한 4개의 변수
	int minCol = object_on_original.cols - 1;
	int minRow = 0;
	int maxCol = 0;
	int maxRow = 0;

	//오른쪽 부분의 gradient를 구하기 위한 변수
	bool isPlusGradient;

	//gradient를 구하기 위한 과정
	//현재는 반복문을 2번 사용해야 하지만 추후에 여유가 되면 개선할 예정
	for (int i = object_on_original.rows - 1; i >= 0; i--) {
		for (int j = object_on_original.cols - 1; j >= 0; j--) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {			
				//minCol이나 maxCol이 갱신되었을 때 minRow와 maxRow 초기화
				//minCol과 j가 같을수 있고 이때 값 갱신을 막는다. 때문에 minMax 사용을 포기함
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

	//두 좌표를 비교해서 gradient를 구한다.
	//두 좌표가 같을때는 Row는 구별하고 Col은 똑같다 -> 따로 분류할 필요 없음
	if (minRow >= maxRow)
		isPlusGradient = true;
	else
		isPlusGradient = false;


	//gradient를 기반으로 좌표 설정
	//isPlusGradient가 true인 경우에는 max는 계속 갱신하고 min은 가장 처음 좌표를 유지
	//false인 경우에는 반대로
	for (int i = object_on_original.rows - 1; i >= 0; i--) {
		for (int j = object_on_original.cols - 1; j >= 0; j--) {
			if (object_on_original.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
				//minCol이나 maxCol이 갱신되었을 때 minRow와 maxRow 초기화
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

	//이미지 수정
	//좌표 순서 : 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
	//행렬에서 성분은 (row, col)이지만 좌표는 (x,y)이고 x는 가로 y는 세로 방향이다.
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

	//원 생성
	/*circle(object_on_original, Point2f(topCol, 100), 3, Scalar(0, 0, 255), 3);
	circle(object_on_original, Point2f(minCol, minRow), 3, Scalar(0, 255, 0), 3);
	circle(object_on_original, Point2f(bottomCol, originalCutImage.rows - 1 + 100), 3, Scalar(255, 0, 0), 3);
	circle(object_on_original, Point2f(maxCol, maxRow), 3, Scalar(255, 255, 255), 3);*/
	

	outputPts.push_back(Point2f(topCol, 100));
	outputPts.push_back(Point2f(minCol, 100));
	outputPts.push_back(Point2f(bottomCol, originalCutImage.rows - 1 + 100));
	outputPts.push_back(Point2f(minCol, originalCutImage.rows - 1 + 100));
	imshow("object_on_original2", object_on_original);


	//변환행렬 생성
	Mat M = getPerspectiveTransform(inputPts, outputPts);
	Mat anotherM;
	warpPerspective(object_on_original, anotherM, M, object_on_original.size(), INTER_CUBIC);
	imshow("M", anotherM);

	//object_on_original과 M을 결합한 Mat 생성 및 값 삽입
	//빨간점과 파란점을 이은 직선을 경계로 두 이미지를 합친다.
	float line = (float)(topCol - bottomCol) / (float)originalCutImage.rows;
	Mat combineImg(originalCutImage.rows, minCol, CV_8UC3);
	for (int i = 0; i < originalCutImage.rows; i++) {
		for (int j = 0; j < minCol; j++) {
			//topCol이 bottomCol보다 큰 경우 빼주고 반대인 경우에는 더한다.
			if (topCol >= bottomCol) {
				if (i < topCol - (int)(line * i))
					combineImg.at<Vec3b>(i,j) = object_on_original.at<Vec3b>(i + 100, j);
				else
					combineImg.at<Vec3b>(i, j) = anotherM.at<Vec3b>(i + 100, j);
			}
			else {
				if (i < topCol + (int)(line * i))
					combineImg.at<Vec3b>(i, j) = object_on_original.at<Vec3b>(i + 100, j);
				else
					combineImg.at<Vec3b>(i, j) = anotherM.at<Vec3b>(i + 100, j);
			}		
		}
	}

	imshow("combine", combineImg);

	//결과를 저장할 mat 생성 후 데이터 옮기기
	//잘린 original과 object_on_original를 합쳐야 한다.
	//분기문을 넣은 이유는 original_image.cols - originalCutImage.cols가 0이면 오류가 발생하기 때문이다.
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
