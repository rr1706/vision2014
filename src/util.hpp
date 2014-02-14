#ifndef UTIL_HPP
#define UTIL_HPP
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <deque>

enum ProcessingMode {
    SA,
    DEMO
};

enum InputSource {
    CAMERA = 0,
    IMAGE = 1,
    VIDEO = 2
};

enum TargetCase {
    NONE = 0,
    ALL = 1,
    RIGHT = 2,
    LEFT = 3,
    ALL_INVERTED = ALL // TODO make this actually have an effect
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

namespace Target {
    enum Type {
        STATIC,
        DYNAMIC
    };

    struct Target {
        Type type;
        double realDistance;
        double planeDistance;
        cv::Moments moment;
        cv::Point2f massCenter;
        cv::Point2i rectCenter;
        cv::Rect boundRect;
        cv::RotatedRect areaRect;
        std::vector<cv::Point2f> corners;
    };
}

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
    std::string str(Part part);
}

namespace Window {
    void print(std::string str, cv::Mat &output, cv::Point pos = cv::Point(5, 15), float size = 1.0);
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
    std::string str(WindowMode mode);
    void print(WindowMode mode, cv::Mat &output);
}

typedef bool (*ContourConstraint)(std::vector<cv::Point>);

struct BallTest {
    std::string name;
    ContourConstraint check;
};

template<class T>
T square ( T x )
{
    return ( x * x );
}

template<class T>
double distance ( const T x1, const T y1, const T x2, const T y2 )
{
    return ( sqrt ( square ( x1 - x2 ) + square ( y1 - y2 ) ) );
}

float distance ( const cv::Point p1, const cv::Point p2 );

std::string xy(const cv::Point2d p1);

std::string xyz(const cv::Point3d p1);

bool isAlmostSquare ( const double ratio );

bool isExtraLong(const double ratio);

double inchesToMeters(const double inches);



/**
 * @brief passesTests Tests if contour passes list of tests
 * @param contour Contour detected in image
 * @param ballTests Tests to run on the contour
 * @return empty string if the contour passes, otherwise return name of failed test
 */
const std::string passesTests(
        std::vector<cv::Point> &contour,
        std::vector<BallTest> &ballTests);

/**
 * @brief getSuccessfulContours Check whether all contours pass all tests
 * @param contours All detected contours
 * @param ballTests All tests to check
 * @return vector of contours that passed all tests
 */
std::vector<std::vector<cv::Point> > getSuccessfulContours(
        std::vector<std::vector<cv::Point> > &contours,
        std::vector<BallTest> &ballTests);

void T2B_L2R(std::vector<cv::Point2f> pt);

template<class List>
cv::Point2d getMedianPoint(List &items, int start, int end);

template<class ArrayOfPoints>
void T2B_L2R(ArrayOfPoints pt);

void applyText(std::vector<std::string> &text, cv::Point startPos, cv::Mat &img);

/**
  * Sort the targets left to right.
  */
void sortTargets(std::vector<Target::Target> &targets);

#endif // UTIL_HPP
