#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>

#include "util.hpp"

extern WindowMode::WindowMode displayMode;
extern std::string windowName;
extern int gray_min;
extern int gray_max;
extern Thresh::Part currentThreshold;
extern unsigned char ballHueMin, ballHueMax, ballSatMin, ballSatMax, ballValMin, ballValMax;
extern cv::Mat kernel0, kernel1, kernel2;
extern int dilations, accuracy;
const int CAMERA_COUNT = 3;
const int TARGET_COUNT = 8;
extern int contourMinArea;
extern float Tan_FOV_Y_Half; // evil magic number that works, not actually tan(fov_y / 2)
extern cv::Size winSize, zeroZone;
extern cv::TermCriteria criteria;
extern ProcessingMode procMode;
extern unsigned int ballMinArea, ballSidesMin;
extern double ballRatioMin, ballRatioMax;
const cv::Size resolution = cv::Size(852, 480);
extern bool SAVE_IMAGES, SAVE_LOGS;
const bool FLIP_IMAGE = true;
const bool FLIP_IMAGE_CAMERA = 2;

#endif // CONFIG_HPP
