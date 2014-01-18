#ifndef UTIL_HPP
#define UTIL_HPP
#include <opencv2/core/core.hpp>
#include <string>
#include <sstream>

enum Mode {
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

std::string xyz(const cv::Point p1)
{
    std::stringstream ret;
    ret << "x=" << p1.x << ",y=" << p1.y << "";
    return ret.str();
}

bool isAlmostSquare ( const double ratio )
{
    return ( ratio < 3 && ratio > 0.5 );
}

bool isExtraLong(const double ratio)
{
    return ratio > 10 || ratio < 0.1;
}

#endif // UTIL_HPP
