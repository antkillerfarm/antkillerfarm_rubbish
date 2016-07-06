#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	vector<Mat> channels;
	vector<Mat> channels0;
	vector<Mat>::iterator it;

	Mat img = imread("../../img/1.jpeg");
	Mat img0;

	split(img,channels);
	it = channels0.begin();
	it = channels0.insert(it, channels[0]);
	it = channels0.insert(it, channels[1]);
	it = channels0.insert(it, channels[2]);
	merge(channels0,img0);

#if 0
	Size size = img.size();
	cout << "Mat Size:" << hex << size.width << ", " << size.height << endl;
#endif

#if 1
	cout << hex << static_cast<int>(img.at<Vec3b>(0,0)[0]) << " ";
	cout << hex << static_cast<int>(img.at<Vec3b>(0,0)[1]) << " ";
	cout << hex << static_cast<int>(img.at<Vec3b>(0,0)[2]) << endl;
	cout << hex << static_cast<int>(channels[0].at<uchar>(0,0)) << endl;
	cout << hex << static_cast<int>(channels[1].at<uchar>(0,0)) << endl;
	cout << hex << static_cast<int>(channels[2].at<uchar>(0,0)) << endl;
#endif

#if 0
	namedWindow("Picture1");  
	imshow("Picture1", img);
	moveWindow("Picture1", 0, 0);

	namedWindow("Picture2");
	imshow("Picture2", img0);
	moveWindow("Picture2", 800, 0);

	waitKey(6000);
#endif
	return 0;
}
