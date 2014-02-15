/* These tools are provided open source AS-IS   */
/* The purpose is to create a toolbox for using */
/* OpenNI depth sensors such as the Kinect or   */
/* Asus Xtion Pro Live. These functions I made  */
/* are extremely early prototype functions and  */
/* are intended for learning purposes. ThankYou */
/*                                              */
/* Emerson O'Hara - Team 1706                   */

#include "opencv2/opencv.hpp"

using namespace cv;

#ifndef DEPTHCONVERSION_H
#define DEPTHCONVERSION_H
extern Mat depthTo8Bit(Mat& c);
extern Mat showDepthAtMouseInColor(Mat& c, const String windowName, double fontSize = 0.5, Scalar color = Scalar(255,0,255));
extern void dualWindowInit(const String depthWindow, const String bgrWindow);
extern void dualImShow(Mat image1, Mat image2, const String window1, const String window2);
#endif // DEPTHCONVERSION_H
