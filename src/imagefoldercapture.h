#ifndef IMAGEFOLDERCAPTURE_H
#define IMAGEFOLDERCAPTURE_H

#include <opencv2/highgui/highgui.hpp>
#include <string>

enum {
    IFC_CAP_PROP_INDEX = 1706001,
    IFC_CAP_PROP_LOOP
};

class ImageFolderCapture : public cv::VideoCapture
{
public:
    ImageFolderCapture();
    ImageFolderCapture(const std::string& folder_path);
    ImageFolderCapture(int) = delete;

    virtual bool open(const cv::string &filename);
    virtual bool open(int device);
    virtual bool isOpened() const;
    void release();

    bool grab();
    bool retrieve(cv::Mat &image, int channel);
    ImageFolderCapture& operator>>(cv::Mat& image);
    bool read(cv::Mat &image);

    bool set(int propId, double value);
    double get(int propId);
protected:
    std::string folder_path;
    int image_index;
    cv::Mat image_buffer;
    bool loop;
};

#endif // IMAGEFOLDERCAPTURE_H
