#include <util.hpp>
#include <iostream>

std::string Thresh::str(Part part) {
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
    default:
        throw;
    }
}

void Window::print(std::string str, cv::Mat &output, cv::Point pos, float size) {
    putText(output, str, pos, CV_FONT_HERSHEY_PLAIN, size, cv::Scalar(255,0,255), 1, 8, false);
}

std::string WindowMode::str(WindowMode mode) {
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
    default:
        throw;
    }
}

void WindowMode::print(WindowMode mode, cv::Mat &output) {
    char out[255];
    sprintf(out, "%d - %s", mode, str(mode).c_str());
    Window::print(std::string(out), output, cv::Point(5, 15));
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

const std::string passesTests(
        std::vector<cv::Point> &contour,
        std::vector<BallTest> &ballTests)
{
    for (auto &test : ballTests) {
        if (!test.check(contour)) {
            return test.name;
        }
    }
    return "";
}

std::vector<std::vector<cv::Point> > getSuccessfulContours(
        std::vector<std::vector<cv::Point> > &contours,
        std::vector<BallTest> &ballTests)
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
    throw;
}

void T2B_L2R(std::vector<cv::Point2f> pt)
{
    int temp_x;
    int temp_y;

    int i, swapped;

    do {
        swapped = 0;
        for (i = 1; i < 4; i++)
        {
            if (pt[i-1].y > pt[i].y)
            {
                temp_x = pt[i-1].x;
                temp_y = pt[i-1].y;
                pt[i-1].x = pt[i].x;
                pt[i-1].y = pt[i].y;
                pt[i].x = temp_x;
                pt[i].y = temp_y;
                swapped = 1;
            }
        }
    }
    while (swapped == 1);

    /// Make sure top two points are left to right
    if (pt[0].x > pt[1].x)
    {
        temp_x = pt[0].x;
        temp_y = pt[0].y;
        pt[0].x = pt[1].x;
        pt[0].y = pt[1].y;
        pt[1].x = temp_x;
        pt[1].y = temp_y;
    }

    /// Make sure bottom two points are left to right
    if (pt[2].x > pt[3].x)
    {
        temp_x = pt[2].x;
        temp_y = pt[2].y;
        pt[2].x = pt[3].x;
        pt[2].y = pt[3].y;
        pt[3].x = temp_x;
        pt[3].y = temp_y;
    }
}

void applyText(std::vector<std::string> &text, cv::Point startPos, cv::Mat &img)
{
    int x = startPos.x;
    int y = startPos.y;
    std::vector<std::string>::const_iterator iterator;
    for (iterator = text.begin(); iterator != text.end(); ++iterator) {
        putText(img, (*iterator).c_str(), cv::Point(x, y), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, cv::Scalar(255,0,255),1,8,false);
        y += 20;
    }
}

void sortTargets(std::vector<Target::Target> &targets)
{
    if (targets.size() <= 1) return;
    bool changed = true;
    while (changed) {
        changed = false;
        for (uint i = 0; i < targets.size(); i++) {
            Target::Target target = targets[i];
            if (i + 1 != targets.size()) {
                Target::Target nextTarget = targets[i + 1];
                if (target.massCenter.x > nextTarget.massCenter.x) {
                    targets[i + 1] = target;
                    targets[i] = nextTarget;
                    changed = true; // if change was made reloop again
                }
            }
        }
    }
}
