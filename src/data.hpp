#ifndef DATA_HPP
#define DATA_HPP
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "util.hpp"
#include "solutionlog.hpp"
#ifdef USE_V4L2
#include "../lib/Webcam.hpp"
#endif

struct ThreadData {
    cv::VideoCapture camera;
#ifdef USE_V4L2
    Webcam* v4l2Cam;
    CameraFrame camFrame;
#endif
    cv::Mat image, original, targetDetect, dst;
    std::vector<Target::Target> targets;
    std::vector<Target::Target> staticTargets;
    std::vector<Target::Target> dynamicTargets;
    TargetCase pairCase = NONE;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point lastImageWrite;
    int imageWriteIndex = 0;
    int id = 0;
    SolutionLog ballLog, targetLog;
    double distanceToBall, angleToBall, ballHeading, ballVelocity;
    int ballArea = 0;
    double robotISA = -99;
};

const struct {
    cv::Point2d fieldOfView;
} cameraInfo = {{111.426, 79}};

const struct {
    double ballWidth;
} fieldData = {0.6096};

#endif // DATA_HPP
