#ifndef UTIL_HPP
#define UTIL_HPP

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
    return ( sqrt ( square ( p1.x - p2.x ) + square ( p1.y - p2.y ) ) );
}

#endif // UTIL_HPP
