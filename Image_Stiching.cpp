#include <iostream>
#include <vector>
#include <opencv2\stitching\stitcher.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char* argv[]) {
	const Mat input1 = imread("image1.jpg", 1);
	const Mat input2 = imread("image2.jpg", 1);

	cv::SiftFeatureDetector detector;

	// find the keypoints for both the images
	vector<KeyPoint> keypoints1;
	detector.detect(input1, keypoints1);
	Mat output1;
	drawKeypoints(input1, keypoints1, output1);
	imwrite("sift_result1.jpg", output1);

	vector<KeyPoint> keypoints2;
	detector.detect(input2, keypoints2);
	Mat output2;
	drawKeypoints(input2, keypoints2, output2);
	imwrite("sift_result2.jpg", output2);

	// Extract the descriptors
	SiftDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matches;
	Mat img_matches;

	extractor.compute(input1, keypoints1, descriptors1);
	extractor.compute(input2, keypoints2, descriptors2);
	matcher.match(descriptors1, descriptors2, matches);

	//show result
	drawMatches(input1, keypoints1, input2, keypoints2, matches, img_matches);
	imwrite("matches.jpg", img_matches);

	Mat output_pano;
	vector<Mat> imgs;
	imgs.push_back(input1);
	imgs.push_back(input2);

	Stitcher stitcher = Stitcher::createDefault();
	Stitcher::Status sts = stitcher.stitch(imgs, output_pano);

	imwrite("stitched.jpg", output_pano);

	cv::waitKey();
	return 0;
}
