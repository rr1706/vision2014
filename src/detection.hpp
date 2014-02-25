#ifndef DETECTION_HPP
#define DETECTION_HPP

#include "data.hpp"

void targetDetection(ThreadData &data);

void ballDetection(ThreadData &data);

void robotDetection(ThreadData &data);

extern std::vector<BallTest> ballTests;

#endif // DETECTION_HPP
