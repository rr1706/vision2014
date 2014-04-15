#include "imagefoldercapture.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <err.h>


using namespace std;
using namespace cv;

ImageFolderCapture::ImageFolderCapture()
{
}

ImageFolderCapture::ImageFolderCapture(const string& folder_path)
{
    bool result = open(folder_path);
    if (!result)
        throw;
}

bool ImageFolderCapture::open(const string &filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) == -1) {
        err(1, "Error loading directory");
        return false;
    }
    char* exampleImage = new char[filename.length() + 15];
    snprintf(exampleImage, filename.length() + 15, "%s/raw_img_0.png", filename.c_str());
    if (stat(exampleImage, &buf) == -1) {
        delete [] exampleImage;
        err(1, "Error loading image");
        return false;
    }
    delete [] exampleImage;

    this->folder_path = filename;
    this->image_index = 0;
    return true;
}

bool ImageFolderCapture::open(int)
{
    return false;
}

bool ImageFolderCapture::isOpened() const
{
    return !this->folder_path.empty();
}

void ImageFolderCapture::release()
{
    this->folder_path.clear();
}

static const int MAX_FILENAME_LENGTH = 255;
static const char* IMAGE_NAME_FORMAT = "%s/raw_img_%d.png";

bool ImageFolderCapture::grab()
{
    char filename[MAX_FILENAME_LENGTH];
    snprintf(filename, MAX_FILENAME_LENGTH, IMAGE_NAME_FORMAT, this->folder_path.c_str(), image_index++);
    struct stat buf;
    int result = stat(filename, &buf);
    if (result == -1 && loop) {
        image_index = 0;
        return grab();
    } else if (result == -1) { // Failed to load image file
        perror("Failed to stat image");
        return false;
    }
    Mat image = imread(filename);
    if (image.rows < 1 || image.cols < 1) { // Empty image
        fprintf(stderr, "Error loading image: empty");
        return false;
    }
    this->image_buffer = image;
    return true;
}

bool ImageFolderCapture::retrieve(cv::Mat &image, int)
{
    image = this->image_buffer;
    return true;
}

ImageFolderCapture& ImageFolderCapture::operator>>(cv::Mat& image)
{
    this->grab();
    image = this->image_buffer;
    return *this;
}

bool ImageFolderCapture::read(cv::Mat &image)
{
    image = this->image_buffer;
    return true;
}

bool ImageFolderCapture::set(int propId, double value)
{
    switch (propId)
    {
    case IFC_CAP_PROP_INDEX:
        this->image_index = value;
        break;
    case IFC_CAP_PROP_LOOP:
        this->loop = (value == 1 ? true : false);
        break;
    default:
        return false;
    }
    return true;
}

double ImageFolderCapture::get(int propId)
{
    switch (propId)
    {
    case IFC_CAP_PROP_INDEX:
        return this->image_index;
    case IFC_CAP_PROP_LOOP:
        return this->loop ? 1 : 0;
    default:
        return 0.0;
    }
}
