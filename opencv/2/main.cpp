#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // 读入一张图片（游戏原画）  
    Mat img=imread("../../img/1.jpeg");  
    // 创建一个名为 "游戏原画"窗口  
    namedWindow("游戏原画");  
    // 在窗口中显示游戏原画  
    imshow("游戏原画",img);  
    // 等待6000 ms后窗口自动关闭  
    waitKey(6000); 

    return 0;
}
