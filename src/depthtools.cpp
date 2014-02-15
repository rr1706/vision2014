/* These tools are provided open source AS-IS   */
/* The purpose is to create a toolbox for using */
/* OpenNI depth sensors such as the Kinect or   */
/* Asus Xtion Pro Live. These functions I made  */
/* are extremely early prototype functions and  */
/* are intended for learning purposes. ThankYou */
/*                                              */
/* Emerson O'Hara - Team 1706                   */

#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "depthtools.h"

using namespace cv;
using namespace std;
static Scalar intensity;
static double depth_distance;
static char str[50];
static int a, b;

void CallBackFunc(int event, int x, int y, int, void*)
{
    if ( event == EVENT_MOUSEMOVE )
    {
        a = x;
        b = y;
    }
}
void dualWindowInit(const String window1, const String window2)
{
    namedWindow(window1, WINDOW_AUTOSIZE);
    namedWindow(window2, WINDOW_AUTOSIZE);
    moveWindow(window1,64,24);
    moveWindow(window2,704,24);
}
void dualImShow(Mat image1, Mat image2, const String window1, const String window2)
{
    imshow(window1, image1);
    imshow(window2, image2);
}
Mat depthTo8Bit(Mat& depthImage)
{
    depthImage -= 512;
    depthImage.convertTo(depthImage,CV_8UC1, 0.0625);
    return depthImage;
}
Mat showDepthAtMouseInColor(Mat& depthImage, const String depthWindow, double fontSize, Scalar color)
{
    setMouseCallback(depthWindow, CallBackFunc, NULL);
    intensity = depthImage.at<uchar>(b,a);
    depth_distance = 2*0.311731*(intensity[0])+19.8902;

    if(depth_distance > 19.8902 && depth_distance <= 178.7)
    {
        sprintf(str, "%.2fin", depth_distance);
    }
    else if(depth_distance > 178.7)
    {
        sprintf(str, "> 178.7in");
    }
    else if(depth_distance <= 19.8902)
    {
        sprintf(str, "Error");
    }
    cvtColor(depthImage,depthImage,CV_GRAY2BGR);
    circle(depthImage,Point(a,b),2,color);
    putText(depthImage, str,Point(a,b),FONT_HERSHEY_SIMPLEX,fontSize,color);
    return depthImage;
}
