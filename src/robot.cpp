#include <iostream>

#include "detection.hpp"
#include "config.hpp"
#include "data.hpp"

using namespace std;
using namespace cv;

static vector<ThresholdDataHSV> robotBumpers = {
    {115, 150, 116, 255, 100, 255}, // RED
    {89, 187, 59, 164, 132, 255} // BLUE
};

static vector<BallTest> testsRobots = {
    {"area", [](vector<Point> contour) {
         return contourArea(contour) > 500;
     }},
    {"l", [](vector<Point> contour) {
         Rect boundRect = boundingRect(contour);
         double contourRatio = static_cast<double>(contourArea(contour)) / boundRect.area();
         if (contourRatio < 0.3 && contourRatio > 0.1) printf("Successful ratio is %f.\n", contourRatio);
         return contourRatio < 0.3 && contourRatio > 0.1;
     }}
};

void robotDetection(ThreadData &data)
{
    cvtColor(data.image, data.image, CV_BGR2RGB);
    if (displayMode == WindowMode::RAW) imshow(windowName, data.image);
    cvtColor(data.image, data.image, CV_RGB2HSV);
    Mat threshOutput = Mat::zeros(data.image.rows, data.image.cols, CV_8U), threshDest;
    for (ThresholdDataHSV &thresh : robotBumpers) {
        inRange(data.image, Scalar(thresh.h_min, thresh.s_min, thresh.v_min),
                Scalar(thresh.h_max, thresh.s_max, thresh.v_max), threshDest);
        threshOutput += threshDest;
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
    for (vector<Point> &contour : succeededContours) {
        Rect boundRect = boundingRect(contour);
        rectangle(passing, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0));
    }
    if (displayMode == WindowMode::FINAL) imshow(windowName, passing);
}
