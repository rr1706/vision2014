#ifndef DEPTHLOGGER_H
#define DEPTHLOGGER_H

#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "solutionlog.hpp"

struct DepthResults {
    cv::Point2f location;
    double distance, angle, rotation, velocity, radius;
};

class DepthLogger
{
public:
    DepthLogger(cv::VideoCapture &camera);
    int start();
private:
    cv::VideoCapture camera;
    cv::Mat calibrate;
    cv::Point2d lastBallPosition;
    DepthResults process(cv::Mat &depth, cv::Mat &color);
    SolutionLog log;
    clock_t lastFrameTime, startTime;
    double timeSinceLastFrame(), timeSinceStart();
};

#endif // DEPTHLOGGER_H
