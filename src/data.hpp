#ifndef DATA_HPP
#define DATA_HPP
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "util.hpp"
#include "solutionlog.hpp"

struct ThreadData {
    cv::VideoCapture camera;
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
};

#endif // DATA_HPP
