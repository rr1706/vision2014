#include <unistd.h>
#include <stdio.h>
#include <vector>
#include "depthlogger.h"
#include "util.hpp"
#include "imagewriter.h"
#include "depthtools.h"
#include "data.hpp"
#include "detection.hpp"
using namespace std;
using namespace cv;

static const char* calibrateFile = "./calibrate.png";
static int lowThreshold = 4;
static int kernelSize = 3;
static int cannyRatio = 3;
static int ballAccuracy = 3;

DepthLogger::DepthLogger(VideoCapture &camera)
    : camera(camera), startTime(clock())
{
    if (access(calibrateFile, R_OK) == 0) { // load calibrate image if it exists
        calibrate = imread(calibrateFile, CV_LOAD_IMAGE_GRAYSCALE);
    }
}

void validateImage(Mat &image)
{
    if (image.rows == 0 || image.cols == 0) {
        cerr << "[DepthLogger] Bad image data! (Width|Height is 0)" << endl;
        abort();
    }
}

int DepthLogger::start()
{
    namedWindow("ASUS", WINDOW_NORMAL);
    ImageWriter writer(true, 1.0, "depth");
    log.open(writer.dirname + "/ballpositions.csv", {"frame", "time", "image", "pos_px_x", "pos_px_y", "distance", "rotation", "radius"});
    Mat depth, color;
    for (int frame = 0;; frame++) {
        camera.grab();
        camera.retrieve(depth, CV_CAP_OPENNI_DEPTH_MAP);
        camera.retrieve(color, CV_CAP_OPENNI_BGR_IMAGE);
        validateImage(depth);
        Mat eightBit = depthTo8Bit(depth);
        imshow("ASUS", eightBit);
        writer.writeImage(depth, "", false);
        writer.writeImage(eightBit, "_8bit", true);
        DepthResults results = process(depth, color);
        log.log("frame", frame).log("time", timeSinceStart()).log("image", writer.imageIndex);
        log.log("pos_px_x", results.location.x).log("pos_px_y", results.location.y);
        log.log("distance", results.distance).log("rotation", results.rotation).log("radius", results.radius);
        lastFrameTime = clock();
        char key = waitKey(30);
        switch (key) {
        case 'T':
            lowThreshold++;
            break;
        case 't':
            lowThreshold--;
            break;
        case 'r':
            cannyRatio--;
            break;
        case 'R':
            cannyRatio++;
            break;
        case 27:
            goto end;
        }
    }
    end:
    camera.release();
    return 0;
}

double DepthLogger::timeSinceLastFrame()
{
    return static_cast<double>(lastFrameTime - clock()) / CLOCKS_PER_SEC;
}

double DepthLogger::timeSinceStart()
{
    return static_cast<double>(startTime - clock()) / CLOCKS_PER_SEC;
}

DepthResults DepthLogger::process(Mat &depth, cv::Mat &color)
{
    int imageWidth = depth.cols, imageHeight = depth.rows;
    Mat dst, thresholded, detectedEdges, calibrated = depth.clone();//calibrate - depth - 1;
    threshold(calibrated, thresholded, 1, 255, CV_THRESH_BINARY_INV);
    thresholded = depth - thresholded;
    // Create a matrix of the same type and size as depth_mat (for dst)
    dst.create(thresholded.size(), thresholded.type());
    // Reduce noise with a kernel 3x3
    blur(thresholded, detectedEdges, Size(3, 3));
    // Canny detector
    Canny(detectedEdges, detectedEdges, lowThreshold, lowThreshold * cannyRatio, kernelSize);
    // Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);
    depth.copyTo(dst, detectedEdges);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    imshow("DST", dst);
    cvtColor(color, color, CV_BGR2RGB);
    Mat rgbColorMat = color.clone();
    cvtColor(color, color, CV_RGB2HSV);
    findContours(dst, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));
    vector<vector<Point> > succeededContours = getSuccessfulContours(contours, ballTests);
    vector<vector<Point> > largestContour(static_cast<unsigned int>(1));
    Scalar largestRGB;
    int threshFails = 0;
    for (vector<Point> &contour : succeededContours) {
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);
        Scalar ptColor = color.at<unsigned char>(center);
        Scalar rgbColor = rgbColorMat.at<unsigned char>(center);
        if (!thresholdPixel(ptColor, ballThreshR)) { // TODO more constraints / tests
            threshFails++;
            continue;
        }
        if (largestContour.size() > 0 && largestContour[0].size() > 0) {
            if (contourArea(contour) > contourArea(largestContour[0])) {
                largestContour[0] = contour;
                largestRGB = rgbColor;
            }
        } else {
            largestContour[0] = contour;
            largestRGB = rgbColor;
        }
    }
    if (largestContour[0].size() == 0) {
        largestContour.pop_back();
    }
    cout << "Thresh fails: " << threshFails << endl;
    for (vector<Point> &contour : largestContour) {
        printf("Ball Color R%f G%f B%f", largestRGB[0], largestRGB[1], largestRGB[2]);
        vector<Point> polygon;
        approxPolyDP(contour, polygon, ballAccuracy, true);
        Point2f ballCenterFlat;
        float radius;
        minEnclosingCircle(contour, ballCenterFlat, radius);
        circle(dst, ballCenterFlat, (int)radius, Scalar(0, 0, 255), 2, 8, 0 );
        double diameter = radius * 2.0;
        double ballAngle = (cameraInfo.fieldOfView.x * diameter) / imageWidth;
        double distanceToBall = (1.0 / tan((ballAngle / 2.0) * (CV_PI / 180))) * (fieldData.ballWidth / 2.0);
        line(dst, Point(ballCenterFlat.x, 0), Point(ballCenterFlat.x, imageHeight), Scalar(0, 255, 50));
        line(dst, Point(0, ballCenterFlat.y), Point(imageWidth, ballCenterFlat.y), Scalar(0, 255, 50));
        ballCenterFlat.x = (ballCenterFlat.x - imageWidth / 2); // rebase origin to center
        ballCenterFlat.y = -(ballCenterFlat.y - imageHeight / 2);
        double ballPosXreal = (fieldData.ballWidth * ballCenterFlat.x) / diameter;
        double ballPosYreal = sqrt(square(distanceToBall) - square(ballPosXreal));
        //Point3d ballCenter = Point3d(ballPosXreal, ballPosYreal, distanceToBall);
        Point2d centerXY = Point2d(ballPosXreal, ballPosYreal);
        double angleToBall = acos(centerXY.y / distanceToBall) * (180 / CV_PI);
        Point2d change = lastBallPosition - centerXY;
        double movedDistance = sqrt(square(change.x) + square(change.y)); // real, meters
        double ballVelocity = movedDistance / timeSinceLastFrame(); // meters per second
        double ballHeading = acos(change.x / movedDistance) * (180 / CV_PI);
        lastBallPosition = centerXY;
        // store five points here
        // calculate median of first five for first point
        // calculate median of last five for second point
        return {ballCenterFlat, distanceToBall, angleToBall, ballHeading, ballVelocity, radius};
    }
    return {{0, 0}, 0, 0, 0, 0, 0};
}
