#ifndef UTIL_HPP
#define UTIL_HPP
#include <opencv2/core/core.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <deque>

enum InputSource {
    CAMERA = 0,
    IMAGE = 1,
    VIDEO = 2
};

enum TargetCase {
    NONE = 0,
    ALL = 1,
    RIGHT = 2,
    LEFT = 3
};

enum ColorSystem {
    IR = 0,
    COLOR = 1
};

enum TrackMode {
    TARGET = 0,
    BALL = 1
};

enum TeamColor {
    RED = 0,
    BLUE = 1
};

namespace Thresh {
    enum Part {
        HUE_MIN = 0,
        HUE_MAX = 1,
        SAT_MIN = 2,
        SAT_MAX = 3,
        VAL_MIN = 4,
        VAL_MAX = 5,
        IR_MIN = 6,
        IR_MAX = 7
    };
    std::string str(Part part) {
        switch (part) {
        case HUE_MIN:
            return "Minimum Hue";
        case HUE_MAX:
            return "Maximum Hue";
        case SAT_MIN:
            return "Minimum Saturation";
        case SAT_MAX:
            return "Maximum Saturation";
        case VAL_MIN:
            return "Minimum Value";
        case VAL_MAX:
            return "Maximum Value";
        case IR_MIN:
            return "Minimum IR";
        case IR_MAX:
            return "Maximum IR";
        }
    }
}

namespace Window {
    void print(std::string str, cv::Mat &output, cv::Point pos = cv::Point(5, 15), float size = 1.0) {
        putText(output, str, pos, CV_FONT_HERSHEY_PLAIN, size, cv::Scalar(255,0,255), 1, 8, false);
    }
}

namespace WindowMode {
    enum WindowMode {
        NONE = -1,
        RAW = 1,
        THRESHOLD = 2,
        ERODE = 3,
        DILATE = 4,
        CONTOURS = 5,
        APPROXPOLY = 6,
        PASS = 7,
        FINAL = 0
    };
    std::string str(WindowMode mode) {
        switch (mode) {
        case NONE:
            return "None";
        case RAW:
            return "Raw";
        case THRESHOLD:
            return "Threshold";
        case ERODE:
            return "Erode";
        case DILATE:
            return "Dilate";
        case CONTOURS:
            return "Contours";
        case APPROXPOLY:
            return "ApproxPoly";
        case PASS:
            return "Passed Tests";
        case FINAL:
            return "Final";
        }
    }
    void print(WindowMode mode, cv::Mat &output) {
        char out[255];
        sprintf(out, "%d - %s", mode, str(mode).c_str());
        Window::print(std::string(out), output, cv::Point(5, 15));
    }
}

typedef bool (*ContourConstraint)(std::vector<cv::Point>);

template<class T>
T square ( T x )
{
    return ( x * x );
}

template<class T>
float distance ( const T x1, const T y1, const T x2, const T y2 )
{
    return ( sqrt ( square ( x1 - x2 ) + square ( y1 - y2 ) ) );
}

float distance ( const cv::Point p1, const cv::Point p2 )
{
    return sqrt ( static_cast<double>( square ( p1.x - p2.x ) + square ( p1.y - p2.y ) ) );
}

std::string xy(const cv::Point2d p1)
{
    std::stringstream ret;
    ret << "x=" << p1.x << ",y=" << p1.y;
    return ret.str();
}

std::string xyz(const cv::Point3d p1)
{
    std::ostringstream ret;
    ret.precision(5);
    ret << "x=" << p1.x << ",y=" << p1.y << ",z=" << p1.z;
    return ret.str();
}

bool isAlmostSquare ( const double ratio )
{
    return ( ratio < 3 && ratio > 0.5 );
}

bool isExtraLong(const double ratio)
{
    return ratio > 20 || ratio < 0.01;
}

double inchesToMeters(const double inches)
{
    return inches * 0.0254;
}



/**
 * @brief passesTests Tests if contour passes list of tests
 * @param contour Contour detected in image
 * @param ballTests Tests to run on the contour
 * @return empty string if the contour passes, otherwise return name of failed test
 */
const std::string passesTests(
        std::vector<cv::Point> &contour,
        std::map<const std::string, ContourConstraint> &ballTests)
{
    for (auto &test : ballTests) {
        if (!test.second(contour)) {
            return test.first;
        }
    }
    return "";
}

/**
 * @brief getSuccessfulContours Check whether all contours pass all tests
 * @param contours All detected contours
 * @param ballTests All tests to check
 * @return vector of contours that passed all tests
 */
std::vector<std::vector<cv::Point> > getSuccessfulContours(
        std::vector<std::vector<cv::Point> > &contours,
        std::map<const std::string, ContourConstraint> &ballTests)
{
    std::map<const std::string, int> failedTests;
    std::vector<std::vector<cv::Point> > succeededContours;
    for (auto &contour : contours) {
        std::string failedTest = passesTests(contour, ballTests);
        if (failedTest.empty()) {
            succeededContours.push_back(contour);
        } else {
            failedTests[failedTest]++;
        }
    }
    std::cout << "Failed tests: ";
    for (auto &failure : failedTests) {
        std::cout << failure.first << ":" << failure.second << ", ";
    }
    std::cout << "success:" << succeededContours.size() << std::endl;
    return succeededContours;
}

template<class List>
cv::Point2d getMedianPoint(List &items, int start, int end)
{
    items.push_back(start);
}

#endif // UTIL_HPP
