#include <iostream>

#include "detection.hpp"
#include "config.hpp"
#include "data.hpp"

using namespace std;
using namespace cv;

static char str[255];

static Point2d lastBallPosition = {0, 0};
//deque<Point2d> lastBallPositions;
static auto lastFrameStart = std::chrono::high_resolution_clock::now();
//SolutionLog ballPositions;
static int ballFrameCount = 0;

/**
  * List of tests to be run on the ball to ensure it is valid.
  */
vector<BallTest> ballTests = {
    {"area", [](vector<Point> contour){
         return contourArea(contour) > ballMinArea;
     }},
    {"sides", [](vector<Point> contour){
         vector<Point> polygon;
         approxPolyDP( contour, polygon, accuracy, true );
         return polygon.size() >= ballSidesMin;
     }},
    {"bumper", [](vector<Point> contour) {
         vector<Point> polygon;
         approxPolyDP( contour, polygon, accuracy, true );
         Point2f ballCenterFlat;
         float radius;
         minEnclosingCircle(contour, ballCenterFlat, radius);
         double areaCircle = CV_PI * square(radius);
         int areaContour = contourArea(contour);
         double circleRatio = areaContour / areaCircle;
         cout << "CIRCLE " << areaCircle << " CONTOUR " << areaContour << " RADIUS " << circleRatio << endl;
         return circleRatio > ballRatioMin && circleRatio < ballRatioMax;
     }}
};

void ballDetection(ThreadData &data)
{
    Mat img = data.image;
    int IMAGE_WIDTH = img.cols, IMAGE_HEIGHT = img.rows;
    auto timeNow = std::chrono::high_resolution_clock::now();
    double timeSinceLastFrame = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(timeNow-lastFrameStart).count()) / 1000000000.0;
    Mat dst = img.clone();
    if (displayMode == WindowMode::RAW) {
        WindowMode::print(displayMode, dst);
        Window::print("Ratchet Rockers 1706", dst, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, dst);
    }
    cvtColor(img,img, CV_BGR2RGB);
    cvtColor(img, img, CV_BGR2HSV);
    // Threshold image to
    inRange(img, Scalar(ballHueMin, ballSatMin, ballValMin), Scalar(ballHueMax, ballSatMax, ballValMax), img);
    if (displayMode == WindowMode::THRESHOLD) {
        Mat thresh = img.clone();
        cvtColor(thresh, thresh, CV_GRAY2RGB);
        WindowMode::print(displayMode, thresh);
        int curThreshVal;
        switch (currentThreshold) {
        case Thresh::HUE_MIN:
            curThreshVal = ballHueMin;
            break;
        case Thresh::HUE_MAX:
            curThreshVal = ballHueMax;
            break;
        case Thresh::SAT_MIN:
            curThreshVal = ballSatMin;
            break;
        case Thresh::SAT_MAX:
            curThreshVal = ballSatMax;
            break;
        case Thresh::VAL_MIN:
            curThreshVal = ballValMin;
            break;
        case Thresh::VAL_MAX:
            curThreshVal = ballValMax;
            break;
        default:
            abort();
        }
        sprintf(str, "%s: %d", Thresh::str(currentThreshold).c_str(), curThreshVal);
        putText(thresh, str, Point(5, 30), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,255), 1, 8, false);
        Window::print("Ratchet Rockers 1706", thresh, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, thresh);
    }
    // Get rid of remaining noise
    morphologyEx(img, img, MORPH_OPEN, kernel0, Point(-1, -1), dilations);
    if (displayMode == WindowMode::DILATE) {
        Mat dilate = img.clone();
        cvtColor(dilate, dilate, CV_GRAY2RGB);
        WindowMode::print(displayMode, dilate);
        sprintf(str, "Dilations: %d", dilations);
        putText(dilate, str, Point(5, 30), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,255), 1, 8, false);
        Window::print("Ratchet Rockers 1706", dilate, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, dilate);
    }
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
    if (displayMode == WindowMode::CONTOURS || displayMode == WindowMode::APPROXPOLY) {
        Mat contoursImg = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);
        cvtColor(contoursImg, contoursImg, CV_GRAY2RGB);
        for ( unsigned int i = 0; i < contours.size(); i++) {
            vector<Point> contour = contours[i];
            if (contourArea(contour) < contourMinArea) {
                continue;
            }
            if (displayMode == WindowMode::CONTOURS) {
                drawContours(contoursImg, contours, i, Scalar(255, 255, 0));
            }
            if (displayMode == WindowMode::APPROXPOLY) {
                Point2f ballCenterFlat;
                float radius;
                minEnclosingCircle(contour, ballCenterFlat, radius);
                circle(contoursImg, ballCenterFlat, (int)radius, Scalar(0, 0, 255), 2, 8, 0 );
            }
        }
        if (displayMode == WindowMode::CONTOURS) {
            sprintf(str, "Area: %d", contourMinArea);
        } else if (displayMode == WindowMode::APPROXPOLY) {
            sprintf(str, "Accuracy: %d", accuracy);
        }
        WindowMode::print(displayMode, contoursImg);
        putText(contoursImg, str, Point(5, 30), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,255), 1, 8, false);
        Window::print("Ratchet Rockers 1706", contoursImg, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, contoursImg);
    }
    vector<vector<Point> > succeededContours = getSuccessfulContours(contours, ballTests);
    vector<vector<Point> > largestContour(static_cast<unsigned int>(1));
    for (vector<Point> &contour : succeededContours) {
        if (largestContour.size() > 0 && largestContour[0].size() > 0) {
            if (contourArea(contour) > contourArea(largestContour[0])) {
                largestContour[0] = contour;
            }
        } else {
            largestContour[0] = contour;
        }
    }
    if (largestContour[0].size() == 0) {
        largestContour.pop_back();
    }
    if (displayMode == WindowMode::PASS) {
        Mat pass = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);
        cvtColor(pass, pass, CV_GRAY2RGB);
        drawContours(pass, largestContour, 0, Scalar(255, 255, 0));
        WindowMode::print(displayMode, pass);
        Window::print("Ratchet Rockers 1706", pass, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, pass);
    }
    double angleToBall = 0;
    double ballVelocity = 0;
    double distanceToBall = 0;
    double ballHeading = 0;
    for (vector<Point> &contour : largestContour) {
        data.ballArea = contourArea(contour);
        vector<Point> polygon;
        approxPolyDP( contour, polygon, accuracy, true );
        Point2f ballCenterFlat;
        float radius;
        minEnclosingCircle(contour, ballCenterFlat, radius);
        circle( dst, ballCenterFlat, (int)radius, Scalar(0, 0, 255), 2, 8, 0 );
        double diameter = radius * 2.0;
        double ballAngle = (cameraInfo.fieldOfView.x * diameter) / IMAGE_WIDTH;
        distanceToBall = (1.0 / tan((ballAngle / 2.0) * (CV_PI / 180))) * (fieldData.ballWidth / 2.0);
        line(dst, Point(ballCenterFlat.x, 0), Point(ballCenterFlat.x, IMAGE_HEIGHT), Scalar(0, 255, 50));
        line(dst, Point(0, ballCenterFlat.y), Point(IMAGE_WIDTH, ballCenterFlat.y), Scalar(0, 255, 50));
        ballCenterFlat.x = (ballCenterFlat.x - IMAGE_WIDTH / 2); // rebase origin to center
        ballCenterFlat.y = -(ballCenterFlat.y - IMAGE_HEIGHT / 2);
        double ballPosXreal = (fieldData.ballWidth * ballCenterFlat.x) / diameter;
        double ballPosYreal = sqrt(square(distanceToBall) - square(ballPosXreal));
        Point3d ballCenter = Point3d(ballPosXreal, ballPosYreal, distanceToBall);
        Point2d centerXY = Point2d(ballPosXreal, ballPosYreal);
        angleToBall = acos(centerXY.y / distanceToBall) * (180 / CV_PI);
        Point2d change = lastBallPosition - centerXY;
        double movedDistance = sqrt(square(change.x) + square(change.y)); // real, meters
        ballVelocity = movedDistance / timeSinceLastFrame; // meters per second
        ballHeading = acos(change.x / movedDistance) * (180 / CV_PI);
        Point pos = contour[0];
        sprintf(str, "Dia:%.2fpx", diameter);
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        pos.y += 10;
        sprintf(str, "BallAng:%.2fo", ballAngle);
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        pos.y += 10;
        sprintf(str, "Dist:%.2fm", distanceToBall);
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        // Notes on position ball
        // The plane on which XY resides is a top-view basically
        // X is the distance over from the center of the camera
        // Y is the distance from the camera to the ball
        // See connor's engineering notebook for more, page 23
        sprintf(str, "Center:%s", xyz(ballCenter).c_str());
        pos.y += 10;
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        sprintf(str, "Change:%s", xy(change).c_str());
        pos.y += 10;
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        sprintf(str, "Moved:%.2fm Angle:%.2f Velocity:%.2fm/s", movedDistance, angleToBall, ballVelocity);
        pos.y += 10;
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        sprintf(str, "Heading:%.2f", ballHeading);
        pos.y += 10;
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        lastBallPosition = centerXY;
//        if (lastBallPositions.size() > 10) {
//            lastBallPositions.pop_front();
//        }
//        lastBallPositions.push_back(centerXY);
        // store five points here
        // calculate median of first five for first point
        // calculate median of last five for second point
        data.ballLog.log("img_x", ballCenterFlat.x).log("img_y", ballCenterFlat.y);
        data.ballLog.log("rel_x", ballPosXreal).log("rel_y", ballPosYreal);
        data.ballLog.log("distance", distanceToBall).log("rotation", ballHeading).log("velocity", ballVelocity).log("heading", ballHeading);
    }
    line(dst, Point(IMAGE_WIDTH / 2, 0), Point(IMAGE_WIDTH / 2, IMAGE_HEIGHT), Scalar(0, 255, 255));
    line(dst, Point(0, IMAGE_HEIGHT / 2), Point(IMAGE_WIDTH, IMAGE_HEIGHT / 2), Scalar(0, 255, 255));
    if (displayMode == WindowMode::FINAL && procMode == DEMO) {
        WindowMode::print(displayMode, dst);
        Window::print("Ratchet Rockers 1706", dst, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, dst);
    }
    data.distanceToBall = distanceToBall;
    data.angleToBall = angleToBall;
    data.ballHeading = ballHeading;
    data.ballVelocity = ballVelocity;
    lastFrameStart = timeNow;
    ballFrameCount++;
}

