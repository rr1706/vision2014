#include <iostream>

#include "detection.hpp"
#include "config.hpp"
#include "data.hpp"

using namespace std;
using namespace cv;

static vector<ThresholdDataHSV> robotBumpers = {
//    {105, 150, 116, 255, 100, 255}, // RED
//    {6, 10, 2, 47, 70, 218} // BLUE
    {55, 131, 38, 148, 106, 209}
};

static vector<BallTest> testsRobots = {
    {"area", [](vector<Point> contour) {
         return contourArea(contour) > 500;
     }}
};


void robotDetection(ThreadData &data)
{
    cvtColor(data.image, data.image, CV_BGR2RGB);
    if (displayMode == WindowMode::RAW) imshow(windowName, data.image);
    cvtColor(data.image, data.image, CV_BGR2HSV);
    Mat threshOutput = Mat::zeros(data.image.rows, data.image.cols, CV_8U);
    for (ThresholdDataHSV &thresh : robotBumpers) {
        Mat threshDest;
        inRange(data.image, Scalar(thresh.h_min, thresh.s_min, thresh.v_min),
                Scalar(thresh.h_max, thresh.s_max, thresh.v_max), threshDest);
        cv::bitwise_or(threshDest, threshOutput, threshOutput);
    }
    data.image = threshOutput;
    if (displayMode == WindowMode::THRESHOLD) imshow(windowName, data.image);
    morphologyEx(data.image, data.image, MORPH_OPEN, kernel0, Point(-1, -1), dilations);
    if (displayMode == WindowMode::DILATE) imshow(windowName, data.image);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(data.image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
    vector<vector<Point> > succeededContours = getSuccessfulContours(contours, testsRobots);
    Mat passing = Mat::zeros(data.image.rows, data.image.cols, CV_8U), final = Mat::zeros(data.image.rows, data.image.cols, CV_8U);
    cvtColor(passing, passing, CV_GRAY2BGR);
    drawContours(passing, succeededContours, -1, Scalar(255, 0, 0));
    if (displayMode == WindowMode::PASS) imshow(windowName, passing);
    vector<Point> largestContour;
    for (vector<Point> &contour : succeededContours) {
        if (largestContour.size() > 0) {
            if (contourArea(contour) > contourArea(largestContour)) {
                largestContour = contour;
            }
        } else {
            largestContour = contour;
        }
    }
    if (largestContour.size() == 0) {
        // no contour found
        return;
    }
    Moments moment = moments(largestContour, false);
    Point2f massCenter(moment.m10/moment.m00, moment.m01/moment.m00);
    if (massCenter.x < 200 || massCenter.x > 600) {
        // another overlap test, only populate if near center
        data.robotISA = (cameraInfo.fieldOfView.x / data.image.cols) * massCenter.x - 55.5;
        printf("Robot ISA:%.2f MCX:%.2f MCY:%.2f\n", data.robotISA, massCenter.x, massCenter.y);
        circle(final, massCenter, 15, Scalar(255, 255, 255));
        if (displayMode == WindowMode::FINAL) imshow(windowName, final);
    }
}
