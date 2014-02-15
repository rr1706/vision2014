#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <opencv2/highgui/highgui.hpp>
#include "imagewriter.h"

static void mkdir_except(const char* dirname)
{
    if (mkdir(dirname, 0755) == -1) {
        if (errno == EEXIST) return;
        fprintf(stderr, "Failed to create folder '%s' to save images.\n", dirname);
        abort();
    }
}

ImageWriter::ImageWriter(bool createDirectory, double writeInterval, std::string subdir, std::string dirname)
    : writeInterval(writeInterval)
{
    if (dirname.empty())
        dirname = getDirnameNow();
    if (createDirectory)
        mkdir_except(dirname.c_str());
    if (!(subdir == ".")) {
        dirname += "/" + subdir;
        if (createDirectory)
            mkdir_except(dirname.c_str());
    }
}

void ImageWriter::writeImage(cv::Mat &image, std::string suffix, bool incrementIndex)
{
    if (static_cast<double>(lastWrite - clock()) < writeInterval) return;
    lastWrite = clock();
    sprintf(imageName, imageNameFormat, dirname.c_str(), imageIndex, suffix.c_str());
    imwrite(imageName, image);
    if (incrementIndex)
        imageIndex++;
}

std::string getDirnameNow()
{
    char dirname[50];
    time_t rawtime;
    time(&rawtime);
    struct tm* timeinfo = localtime(&rawtime);
    strftime(dirname, 50, "%Y%m%d_%H%M%S", timeinfo);
    return std::string(dirname);
}
