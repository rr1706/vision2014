#ifndef IMAGEWRITER_H
#define IMAGEWRITER_H
#include <opencv2/core/core.hpp>
#include <time.h>
#include <string>

std::string getDirnameNow();

const char* const imageNameFormat = "%s/raw_img_%d%s.png";

class ImageWriter
{
public:
    ImageWriter(bool createDirectory = false, double imageWriteInterval = 1.0, std::string subdir = ".", std::string dirname = "");
    void writeImage(cv::Mat &image, std::string suffix = "", bool incrementIndex = true);
    int imageIndex;
    std::string dirname;
private:
    clock_t lastWrite;
    double writeInterval;
    char imageName[255];
};

#endif // IMAGEWRITER_H
