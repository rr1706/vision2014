#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <map>
#include <vector>
#include <deque>
#include <thread>
#include <atomic>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <QtNetwork>
#include <sys/types.h>
#include <sys/stat.h>
#include "util.hpp"
#include "solutionlog.hpp"
#include "xyh.hpp"

#define FOV_Y 79//degrees
#define FOV_X 111.426 //degrees
#define STATIC_TARGET_HEIGHT  32.25 //in
#define DYNAMIC_TARGET_HEIGHT 4 //in

// OpenCV Namespace
using namespace cv;
using namespace std;

// for sprintf (grr)
char str[255];

// keys
const char KEY_QUIT = 27;
const char KEY_SAVE = 'w';
const char KEY_SPEED = ' ';

// config
const ProcessingMode procMode = SA;
const InputSource mode = CAMERA;
const int cameraId = 1;
const ColorSystem inputType = IR;
const TrackMode tracking = TARGET;
const string videoPath = "Y400cmX646cm.avi";
// displayImage replaced with WindowMode::NONE
const TeamColor color = RED;
const bool doUdp = true;
const QHostAddress udpRecipient(0xC049EE66);
QUdpSocket udpSocket;
const bool saveImages = true;
const double imageInterval = 1.0; // seconds

// Values for threshold IR
int gray_min = 245;
int gray_max = 255;

// Values for threshold RGB
const int hue_min = 35;
const int hue_max = 90;
const int saturation_min = 10;
const int saturation_max = 255;
const int value_min = 140;
const int value_max = 255;

// Values for threshold ball track
uchar ballHueMin = color == RED ? 115 : 31;
uchar ballHueMax = color == RED ? 150 : 128;
uchar ballSatMin = color == RED ? 116 : 92;
uchar ballSatMax = color == RED ? 255 : 202;
uchar ballValMin = color == RED ? 100 : 0;
uchar ballValMax = color == RED ? 255 : 158;
const uint ballSidesMin = 5; // for a circle
const uint ballMinArea = 250;

// for approxpolydp
int accuracy = 2; //maximum distance between the original curve and its approximation
int contourMinArea = 50;
const float Tan_FOV_Y_Half = 1.46;

const int kern_mat[] = {1,0,1,
                        0,1,0,
                        1,0,1};
const Mat kernel = getStructuringElement(*kern_mat, Size(3,3), Point(-1,-1));
const Size winSize = Size( 5, 5 );
const Size zeroZone = Size( -1, -1 );
const TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

vector<BallTest> ballTests = {
    {"area", [](vector<Point> contour){
         return contourArea(contour) > ballMinArea;
     }},
    {"sides", [](vector<Point> contour){
         vector<Point> polygon;
         approxPolyDP( contour, polygon, accuracy, true );
         return polygon.size() >= ballSidesMin;
     }}
};
auto startTime = std::chrono::high_resolution_clock::now();
auto lastImageWrite = std::chrono::high_resolution_clock::now();

int imageWriteIndex = 0;
int dilations = 1;
WindowMode::WindowMode displayMode = WindowMode::FINAL;

char dirname[255];
time_t rawtime;
struct tm* timeinfo;
const string windowName = "2014";
Thresh::Part currentThreshold = Thresh::HUE_MIN;
const int CHANGE_THRESH = 5;
const int CHANGE_AREA = 20;
const int CHANGE_ACCURACY = 1;
const int CHANGE_DILATE = 1;

const int CAMERA_COUNT = 3;
const int TARGET_COUNT = 8;

const Point3d worldCoordsLeft[] = {
    {10.91666666666667, 10.01041666666667, 0},
    {10.91666666666667, 8.34375, 0},
    {16.08333333333334, 8.34375, 0},
    {16.08333333333334, 10.01041666666667, 0},
    {10.91666666666667, 10.01041666666667, 0},
    {10.91666666666667, 8.34375, 0},
    {16.08333333333334, 8.34375, 0},
    {16.08333333333334, 10.01041666666667, 0}
};

const Point3d worldCoordsRight[] = {
    {10.91666666666667, 10.01041666666667, 0},
    {10.91666666666667, 8.34375, 0},
    {16.08333333333334, 8.34375, 0},
    {16.08333333333334, 10.01041666666667, 0},
    {10.91666666666667, 10.01041666666667, 0},
    {10.91666666666667, 8.34375, 0},
    {16.08333333333334, 8.34375, 0},
    {16.08333333333334, 10.01041666666667, 0}
};

const Point3d worldCoordsBoth[] = {
    {10.91666666666667, 10.01041666666667, 0},
    {10.91666666666667, 8.34375, 0},
    {16.08333333333334, 8.34375, 0},
    {16.08333333333334, 10.01041666666667, 0},
    {10.91666666666667, 10.01041666666667, 0},
    {10.91666666666667, 8.34375, 0},
    {16.08333333333334, 8.34375, 0},
    {16.08333333333334, 10.01041666666667, 0},
    {10.91666666666667, 10.01041666666667, 0},
    {10.91666666666667, 8.34375, 0},
    {16.08333333333334, 8.34375, 0},
    {16.08333333333334, 10.01041666666667, 0},
    {10.91666666666667, 10.01041666666667, 0},
    {10.91666666666667, 8.34375, 0},
    {16.08333333333334, 8.34375, 0},
    {16.08333333333334, 10.01041666666667, 0}
};

struct ThreadData {
    VideoCapture camera;
    Mat image, original;
    vector<Target::Target> targets;
    vector<Target::Target> staticTargets;
    vector<Target::Target> dynamicTargets;
    TargetCase pairCase = NONE;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point lastImageWrite;
    int imageWriteIndex = 0;
    int id = 0;
};

int demo();
int sa();
void targetDetection(ThreadData &data);
void ballDetection(ThreadData &data);

int main()
{
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(dirname,50,"%Y%m%d_%H%M%S",timeinfo);
    if (saveImages && mkdir(dirname, 0755) == -1) {
        cerr << "Failed to create directory " << dirname << endl;
        return 1;
    }
    return procMode == SA ? sa() : demo();
}

int demo()
{
    ThreadData data;
    if (mode == CAMERA) {
        data.camera = VideoCapture(cameraId);
        if (!data.camera.isOpened()) {
            cerr << "Failed to open camera device id:" << cameraId << endl;
            return -1;
        }
//        data.camera.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
//        data.camera.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    } else if (mode == VIDEO) {
        data.camera = VideoCapture(videoPath);
        if (!data.camera.isOpened()) {
            cerr << "Failed to open video path:" << videoPath << endl;
            return -1;
        }
    }
    int currentSave = 0;

    Mat img, inframe;

    if (mode == IMAGE) {
        inframe = imread("raw_img_0.png");
    }
    if (displayMode != WindowMode::NONE) {
        namedWindow(windowName, CV_WINDOW_NORMAL);
    }
    if (mkdir(dirname, 0755) == -1) {
        strcpy(dirname, ".");
    }

    while ( 1 )
    {
        // Start timing a frame (FPS will be a measurement of the time it takes to process all the code for each frame)
        auto start = std::chrono::high_resolution_clock::now();

        if (mode == CAMERA || mode == VIDEO) { // Replaced #if with braced conditional. Modern compiler should have no performance differences.
            // Grab a frame and contain it in the cv::Mat img
            data.camera >> img;
        } else {
            img = inframe.clone();
        }

        //Break out of loop if esc is pressed
        switch (char key = waitKey(10)) {
        case KEY_QUIT:
            return 0;
            break;
        case KEY_SAVE:
            sprintf(str, "raw_img_%d.png", currentSave);
            imwrite(str, img);
            currentSave++;
            break;
        case KEY_SPEED:
            for (int gi = 0; gi < 20; gi++) {
                data.camera >> img;
            }
            break;
        case 'p':
            waitKey();
            break;
        case '-':
            switch (displayMode) {
            case WindowMode::THRESHOLD:
                switch (currentThreshold) {
                case Thresh::HUE_MIN:
                    ballHueMin -= CHANGE_THRESH;
                    break;
                case Thresh::HUE_MAX:
                    ballHueMax -= CHANGE_THRESH;
                    break;
                case Thresh::SAT_MIN:
                    ballSatMin -= CHANGE_THRESH;
                    break;
                case Thresh::SAT_MAX:
                    ballSatMax -= CHANGE_THRESH;
                    break;
                case Thresh::VAL_MIN:
                    ballValMin -= CHANGE_THRESH;
                    break;
                case Thresh::VAL_MAX:
                    ballValMax -= CHANGE_THRESH;
                    break;
                case Thresh::IR_MIN:
                    gray_min -= CHANGE_THRESH;
                    break;
                case Thresh::IR_MAX:
                    gray_max -= CHANGE_THRESH;
                    break;
                }
                break;
            case WindowMode::DILATE:
                dilations -= CHANGE_DILATE;
                break;
            case WindowMode::CONTOURS:
                contourMinArea -= CHANGE_AREA;
                break;
            case WindowMode::APPROXPOLY:
                accuracy -= CHANGE_ACCURACY;
                break;
            default:
                break;
            }
            break;
        case '+':
            switch (displayMode) {
            case WindowMode::THRESHOLD:
                switch (currentThreshold) {
                case Thresh::HUE_MIN:
                    ballHueMin += CHANGE_THRESH;
                    break;
                case Thresh::HUE_MAX:
                    ballHueMax += CHANGE_THRESH;
                    break;
                case Thresh::SAT_MIN:
                    ballSatMin += CHANGE_THRESH;
                    break;
                case Thresh::SAT_MAX:
                    ballSatMax += CHANGE_THRESH;
                    break;
                case Thresh::VAL_MIN:
                    ballValMin += CHANGE_THRESH;
                    break;
                case Thresh::VAL_MAX:
                    ballValMax += CHANGE_THRESH;
                    break;
                case Thresh::IR_MIN:
                    gray_min += CHANGE_THRESH;
                    break;
                case Thresh::IR_MAX:
                    gray_max += CHANGE_THRESH;
                    break;
                }
                break;
            case WindowMode::DILATE:
                dilations += CHANGE_DILATE;
                break;
            case WindowMode::CONTOURS:
                contourMinArea += CHANGE_AREA;
                break;
            case WindowMode::APPROXPOLY:
                accuracy += CHANGE_ACCURACY;
                break;
            default:
                break;
            }
            break;
        case 'h' :
            currentThreshold = Thresh::HUE_MIN;
            break;
        case 'H':
            currentThreshold = Thresh::HUE_MAX;
            break;
        case 's' :
            currentThreshold = Thresh::SAT_MIN;
            break;
        case 'S':
            currentThreshold = Thresh::SAT_MAX;
            break;
        case 'v' :
            currentThreshold = Thresh::VAL_MIN;
            break;
        case 'V':
            currentThreshold = Thresh::VAL_MAX;
            break;
        case 'i':
            currentThreshold = Thresh::IR_MIN;
            break;
        case 'I':
            currentThreshold = Thresh::IR_MAX;
            break;
        default:
            if (key >= '0' && key <= '9') {
                int ikey = key - '0';
                displayMode = static_cast<WindowMode::WindowMode>(ikey);
            }
        }
        double timeSinceWrite = std::chrono::duration_cast<std::chrono::duration<double> >(start-lastImageWrite).count();
        if (saveImages && timeSinceWrite > imageInterval) {
            sprintf(str, "%s/raw_img_%d.png", dirname, imageWriteIndex);
            imwrite(str, img);
            imageWriteIndex++;
            lastImageWrite = start;
        }
        data.image = img;
        switch (tracking) {
        case BALL:
            ballDetection(data);
            break;
        case TARGET:
            targetDetection(data);
            break;
        }
        // Note: img is dirty after running these functions

        /// Stop timing and calculate FPS and Average FPS
        auto finish = std::chrono::high_resolution_clock::now();
        double seconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count()) / 1000000000.0;
        cout << "Processed all image code in " << seconds << " seconds." << endl;
        if (mode == IMAGE) {
            char k = waitKey(); // pause
            if (k >= '0' && k <= '9') {
                stringstream filename;
                filename << "raw_img_" << k << ".png";
                inframe = imread(filename.str());
                continue;
            } else {
                break;
            }
        }
    } /// <---- End of While Loop (ESC has to be pressed to break out of loop) otherwise loop

    /// Destroy all windows and return 0 to end the program
    destroyAllWindows();
    return 0;
}

void runThread(ThreadData *data) {
    data->camera >> data->image;
    data->original = data->image.clone();
    targetDetection(*data);
}

int sa()
{
    ThreadData* threadData[CAMERA_COUNT];
    for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
        threadData[i] = new ThreadData;
        sprintf(str, "%s/cam_%d", dirname, i);
        if (saveImages && mkdir(str, 0755) == -1) {
            cerr << "Failed to create directory for camera: " << str << endl;
            return 1;
        }
        threadData[i]->camera.open(i);
    }
    auto begin = std::chrono::high_resolution_clock::now();
    sprintf(str, "%s/sa.csv", dirname);
    SolutionLog saLog(string(str), {"time", "heading", "x", "y"});
    thread threads[CAMERA_COUNT];
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            threadData[i]->start = start;
            threads[i] = thread(runThread, threadData[i]);
        }
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            if (threads[i].joinable())
                threads[i].join();
        }
        double timeSinceWrite = std::chrono::duration_cast<std::chrono::duration<double> >(start-lastImageWrite).count();
        if (saveImages && timeSinceWrite > imageInterval) {
            for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
                sprintf(str, "%s/cam_%d/raw_img_%d.png", dirname, i, imageWriteIndex);
                imwrite(str, threadData[i]->original);
            }
            imageWriteIndex++;
            lastImageWrite = start;
        }
        int P[CAMERA_COUNT][TARGET_COUNT];
        for (unsigned int pi = 0; pi < CAMERA_COUNT; pi++) {
            for (unsigned int pj = 0; pj < TARGET_COUNT; pj++) {
                P[pi][pj] = -1;
            }
        }
        int R[8];
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            ThreadData data = *threadData[i];

            cout << "Thread " << i << " case: " << data.pairCase << endl;
            if (data.pairCase == LEFT) {
                R[0] = data.dynamicTargets[0].realDistance;
                R[4] = data.staticTargets[0].realDistance;
                P[i][4] = data.staticTargets[0].massCenter.x;
                P[i][0] = data.dynamicTargets[0].massCenter.x;
            } else if (data.pairCase == RIGHT) {
                R[4] = data.dynamicTargets[0].realDistance;
                R[0] = data.staticTargets[0].realDistance;
                P[i][4] = data.staticTargets[0].massCenter.x;
                P[i][0] = data.dynamicTargets[0].massCenter.x;
            } else if (data.pairCase == ALL) {
                R[0] = data.staticTargets[0].realDistance;
                R[4] = data.dynamicTargets[0].realDistance;
                R[5] = data.staticTargets[1].realDistance;
                R[1] = data.dynamicTargets[1].realDistance;
                P[i][0] = data.staticTargets[0].massCenter.x;
                P[i][4] = data.dynamicTargets[0].massCenter.x;
                P[i][5] = data.staticTargets[1].massCenter.x;
                P[i][1] = data.dynamicTargets[1].massCenter.x;
            }
        }
        double xPos, yPos, heading;
        FindXYH(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], P, xPos, yPos, heading);
        double tSnSt = std::chrono::duration_cast<std::chrono::duration<double> >(start-begin).count();
        saLog.log("time", tSnSt).log("heading", heading).log("x", xPos).log("y", yPos).flush();
        // stitching is broken currently, TODO test on odroid
//        Mat pano;
//        vector<Mat> imgs;
//        Stitcher stitcher = Stitcher::createDefault();
//        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
//            imgs.push_back(threadData[i]->original.clone());
//        }
//        Stitcher::Status status = stitcher.stitch(imgs, pano);
//        if (status != Stitcher::OK) {
//            cout << "Error stitching - Code: " <<int(status)<<endl;
//            return -1;
//        }
//        Window::print("Ratchet Rockers 1706", pano, Point(pano.cols - 200, 15));
//        sprintf(str, "xPos %.2f yPos %.2f heading %.2f", xPos, yPos, heading);
//        Window::print(string(str), pano, Point(5, 15));
//        imshow(windowName, pano);
        auto finish = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(finish-start).count();
        cout << "Processed all image code in " << seconds << " seconds." << endl;
        switch (char key = waitKey(30)) {
        case 27:
            return key - 27;
        }
    }
    return 0;
}

void targetDetection(ThreadData &data)
{
    Mat img = data.image;
    int IMAGE_WIDTH = img.cols, IMAGE_HEIGHT = img.rows;
    // Store the original image img to the Mat dst
    Mat dst = img.clone();

    // Convert image from input to threshold method
    if (inputType == IR) {
        cvtColor(img, img, CV_BGR2GRAY);
    } else if (inputType == COLOR) {
        cvtColor(img, img, CV_BGR2HSV);
    }
    if (displayMode == WindowMode::RAW) {
        Mat input = img.clone();
        cvtColor(input, input, inputType == IR ? CV_GRAY2RGB : CV_HSV2RGB);
        Window::print("Ratchet Rockers 1706", input, Point(IMAGE_WIDTH - 200, 15));
        sprintf(str, "Input Mode = %s", inputType == IR ? "IR" : "Color");
        putText(input, str, Point(5,15), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
        imshow(windowName, input);
    }

    // "Threshold" image to pixels in the ranges
    if (inputType == IR) {
        threshold(img, img, gray_min, gray_max, CV_THRESH_BINARY);
    } else if (inputType == COLOR) {
        inRange(img, Scalar(hue_min, saturation_min, value_min), Scalar(hue_max, saturation_max, value_max), img);
    }
    if (displayMode == WindowMode::THRESHOLD) {
        Mat thresh = img.clone();
        cvtColor(thresh, thresh, CV_GRAY2RGB); // binary image at this point
        Window::print("Ratchet Rockers 1706", thresh, Point(IMAGE_WIDTH - 200, 15));
        sprintf(str, "%d - Threshold", displayMode);
        Window::print(string(str), thresh, Point(5, 15));
        int curThreshVal;
        switch (currentThreshold) {
        case Thresh::HUE_MIN:
            curThreshVal = ballHueMin;
            break;
        case Thresh::HUE_MAX:
            curThreshVal = ballHueMax;
            break;
        case Thresh::SAT_MIN:
            curThreshVal = ballSatMin;
            break;
        case Thresh::SAT_MAX:
            curThreshVal = ballSatMax;
            break;
        case Thresh::VAL_MIN:
            curThreshVal = ballValMin;
            break;
        case Thresh::VAL_MAX:
            curThreshVal = ballValMax;
            break;
        case Thresh::IR_MIN:
            curThreshVal = gray_min;
            break;
        case Thresh::IR_MAX:
            curThreshVal = gray_max;
            break;
        }
        sprintf(str, "%s: %d", Thresh::str(currentThreshold).c_str(), curThreshVal);
        Window::print(string(str), thresh, Point(5, 30));
        imshow(windowName, thresh);
    }

    // Get rid of remaining noise
//    dilate(img, img, kernel);
//    erode(img, img, kernel, Point(-1, -1), 2);
    morphologyEx(img, img, MORPH_OPEN, kernel, Point(-1, -1), dilations); // note replaced with open, idk if it will work here
    if (displayMode == WindowMode::DILATE) {
        Mat dilate = img.clone();
        Window::print("Ratchet Rockers 1706", dilate, Point(IMAGE_WIDTH - 200, 15));
        sprintf(str, "%d - Dilate", displayMode);
        putText(dilate, str, Point(5,15), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255,0,255),1,8,false);
        imshow(windowName, dilate);
    }

    // Declare containers for contours and contour heirarchy
    vector<vector<Point> > contours;
    vector<Point2f> corners;
    vector<vector<Point> > Static_Target;
    vector<vector<Point> > Dynamic_Target;
    vector<Vec4i> hierarchy;


    findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<string> statusText;
    int totalContours = contours.size();
    int failedArea = 0;
    int failedHierarchy = 0;
    int failedSides = 0;
    int failedConvex = 0;
    int failedSquare = 0;
    int failedVLarge = 0;
    int success = 0;
    double Image_Heigh_in = 0.0;
    Mat contoursImg = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);
    double R[8] = {0};
    int P[CAMERA_COUNT][TARGET_COUNT];
    for (uint pi = 0; pi < CAMERA_COUNT; pi++) {
        for (uint pj = 0; pj < TARGET_COUNT; pj++) {
            P[pi][pj] = -1;
        }
    }
    vector<Point2d> pixel_coords;
    vector<Point3d> world_coords;
    vector<Target::Target> targets, staticTargets, dynamicTargets;

    // Create a for loop to go through each contour (i) one at a time
    for( unsigned int i = 0; i < contours.size(); i++ )
    {
        vector<Point> contour = contours[i];
        // Very small contours (noise)
        if (contourArea(contour) < contourMinArea) {
            failedArea++;
            continue;
        }
        if (displayMode == WindowMode::CONTOURS) {
            drawContours(contoursImg, contours, i, Scalar(255, 255, 0));
        }
        vector<Point> polygon;
        approxPolyDP( contour, polygon, accuracy, true );
        vector<vector<Point> > contoursDrawWrapper {polygon};
        if (displayMode == WindowMode::APPROXPOLY) {
            drawContours(contoursImg, contoursDrawWrapper, 0, Scalar(255, 255, 0));
        }
        if (false && polygon.size() != 4) {
            failedSides++;
            continue;
        }
        if (!isContourConvex(polygon)) {
            failedConvex++;
            continue;
        }
        Rect boundRect = boundingRect(polygon);
//        rectangle( dst, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0), 2, 8, 0 );
        RotatedRect minRect = minAreaRect( Mat(contour));
        Point2f rect_points[4];
        minRect.points(rect_points);
        for (int j = 0; j < 4; j++)
            line(dst, rect_points[j], rect_points[(j+1)%4], Scalar(0, 255, 0),2, 8);

        // ratio helps determine orientation of rectangle (vertical / horizontal)
        double ratio = static_cast<double>(boundRect.width) / static_cast<double>(boundRect.height);
        if (isAlmostSquare(ratio)) {
            failedSquare++;
            continue; // go to next contour
        } else if (isExtraLong(ratio)) {
            failedVLarge++;
            continue;
        }

        vector<Point2f> localCorners;
        for (int k = 0; k < 4; k++)
        {
            localCorners.push_back(polygon[k]);
        }

        // organize corners
        T2B_L2R(localCorners);

        // Calculate the refined corner locations
        cornerSubPix(img, localCorners, winSize, zeroZone, criteria);
        T2B_L2R(localCorners);
        corners.insert(corners.end(), localCorners.begin(), localCorners.end());

        // test aspect ratio
//        double lengthTop = distance(localCorners[0], localCorners[1]);
//        double lengthBottom = distance(localCorners[2], localCorners[3]);
//        double lengthLeft = distance(localCorners[0], localCorners[2]);
//        double lengthRight = distance(localCorners[1], localCorners[3]);
        success++;
        int centerX = boundRect.x + boundRect.width / 2;
        int centerY = boundRect.y + boundRect.height / 2;
        Point2i center = {centerX, centerY};
        Target::Type targetType = (boundRect.height > boundRect.width * 2) ? Target::STATIC : Target::DYNAMIC;
        double planeDistance, realDistance;

        if (targetType == Target::STATIC) // static target
        {
            Image_Heigh_in = (IMAGE_HEIGHT * STATIC_TARGET_HEIGHT) / boundRect.height;
//            double refinedHeight = distance(localCorners[0], localCorners[2]);
//            double flatHeight = localCorners[2].y - localCorners[0].y;
            double Center_Static_X = (boundRect.x + (boundRect.width / 2)) - (IMAGE_WIDTH/2);
            double Plane_Distance = (Image_Heigh_in) / Tan_FOV_Y_Half;
            double In_Screen_Angle = (FOV_X / IMAGE_WIDTH) * Center_Static_X;
            double Real_Distance = Plane_Distance / (cos(In_Screen_Angle * CV_PI / 180));
            planeDistance = Plane_Distance;
            realDistance = Real_Distance;

            if (Plane_Distance < 5) {
                continue;
            }

            sprintf(str, "BRH:%d", boundRect.height);
            putText(dst, str, center, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            sprintf(str, "PLD:%.2fm %fin", inchesToMeters(Plane_Distance), Plane_Distance);
            putText(dst, str, center + Point2i(0, 15), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            circle(dst, localCorners[0], 5, Scalar(0, 255, 255), 2, 8, 0);
            circle(dst, localCorners[2], 5, Scalar(100, 255, 200), 2, 8, 0);
            //contour is a tall and skinny one
            //save off as static target
            Static_Target.push_back(contours[i]);
        }
        else
        {
            if (Image_Heigh_in == 0.0) // only set with dynamic if there is no value, static is probably more accurate
            Image_Heigh_in = (IMAGE_HEIGHT * DYNAMIC_TARGET_HEIGHT) / boundRect.height;
            double Center_Static_X = (boundRect.x + (boundRect.width / 2)) - (IMAGE_WIDTH/2);
            double Plane_Distance_Dynamic = (Image_Heigh_in) / Tan_FOV_Y_Half;
            double In_Screen_Angle_Dynamic = (FOV_X / IMAGE_WIDTH) * Center_Static_X;
            double Real_Distance_Dynamic = Plane_Distance_Dynamic / (cos(In_Screen_Angle_Dynamic * CV_PI / 180));
            planeDistance = Plane_Distance_Dynamic;
            realDistance = Real_Distance_Dynamic;

            if (Plane_Distance_Dynamic < 5) {
                continue;
            }

            sprintf(str, "BRH:%d", boundRect.height);
            putText(dst, str, center, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            sprintf(str, "PLD:%.2fm", inchesToMeters(Plane_Distance_Dynamic));
            putText(dst, str, center + Point2i(0, 15), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));

            circle(dst, localCorners[0], 5, Scalar(0, 255, 255), 2, 8, 0);
            circle(dst, localCorners[2], 5, Scalar(255, 255, 0), 2, 8, 0);

            //contour is the short and wide, dynamic target
            //save off as dynamic target
            Dynamic_Target.push_back(contours[i]);
        }
        Moments moment = moments(contour, false);
        Point2f massCenter(moment.m10/moment.m00, moment.m01/moment.m00);
        Target::Target target = {targetType, realDistance, planeDistance, moment, massCenter, center, boundRect, minRect};
        targets.push_back(target);
        if (targetType == Target::STATIC) {
            staticTargets.push_back(target);
        } else {
            dynamicTargets.push_back(target);
        }
        if (displayMode == WindowMode::PASS) {
            drawContours(contoursImg, contoursDrawWrapper, 0, Scalar(255, 255, 0));
        }
    }
    if (displayMode == WindowMode::CONTOURS || displayMode == WindowMode::APPROXPOLY || displayMode == WindowMode::PASS) {
        WindowMode::print(displayMode, contoursImg);
        Window::print("Ratchet Rockers 1706", contoursImg, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, contoursImg);
    }
    cout << "Total: " << totalContours << " | Failures Area: " << failedArea << " Hierarchy: " << failedHierarchy <<
            " Sides: " << failedSides << " Convex: " << failedConvex << " Square: " << failedSquare << " VeryLarge: " << failedVLarge << " | Success: " << success << endl;
    sortTargets(targets);
    sortTargets(dynamicTargets);
    sortTargets(staticTargets);
    data.targets = targets;
    data.dynamicTargets = dynamicTargets;
    data.staticTargets = staticTargets;
    TargetCase targetCase = NONE;
    if (dynamicTargets.size() > 0 && staticTargets.size() > 0 && targets.size() == 2
            && staticTargets[0].massCenter.x > dynamicTargets[0].massCenter.x) {
        //case left
        targetCase = LEFT;
        R[0] = dynamicTargets[0].realDistance;
        R[4] = staticTargets[0].realDistance;
        P[0][4] = staticTargets[0].massCenter.x;
        P[0][0] = dynamicTargets[0].massCenter.x;
    } else if (dynamicTargets.size() > 0 && staticTargets.size() > 0 && targets.size() == 2) {
        //case right
        targetCase = RIGHT;
        R[4] = dynamicTargets[0].realDistance;
        R[0] = staticTargets[0].realDistance;
        P[0][4] = staticTargets[0].massCenter.x;
        P[0][0] = dynamicTargets[0].massCenter.x;
    }
    if (staticTargets.size() >= 2 && dynamicTargets.size() >= 2) {
        targetCase = ALL;
        R[0] = staticTargets[0].realDistance;
        R[4] = dynamicTargets[0].realDistance;
        R[5] = staticTargets[1].realDistance;
        R[1] = dynamicTargets[1].realDistance;
        P[0][0] = staticTargets[0].massCenter.x;
        P[0][4] = dynamicTargets[0].massCenter.x;
        P[0][5] = staticTargets[1].massCenter.x;
        P[0][1] = dynamicTargets[1].massCenter.x;
    }
    data.pairCase = targetCase;
    double xPos, yPos, heading;
    FindXYH(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], P, xPos, yPos, heading);
    // Get vectors for world->camera transform
//    solvePnP (world_coords, pixel_coords, cameraMatrix, distortion, rvec, tvec);

    // We need inverse of the world->camera transform (camera->world) to calculate
    // the camera's location
    //    Rodrigues (rvec, rotation_matrix);
    //    Rodrigues (rotation_matrix.t (), camera_rotation_vector);
    //    Mat t = tvec.t ();
    //    camera_translation_vector = -camera_rotation_vector * t;

    string caseStr;
    switch (targetCase) {
    case NONE:
        caseStr = "None";
        break;
    case LEFT:
        caseStr = "Left";
        break;
    case RIGHT:
        caseStr = "Right";
        break;
    case ALL:
        caseStr = "Both";
        break;
    }

    sprintf(str, "Case = %s", caseStr.c_str());
    putText(dst, str,Point(5,45), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    sprintf(str, "Targets S:%lu D:%lu", staticTargets.size(), dynamicTargets.size());
    putText(dst, str,Point(5,60), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    sprintf(str, "Image Height %dpx %.2fin", IMAGE_HEIGHT, Image_Heigh_in);
    putText(dst, str,Point(5,75), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    sprintf(str, "Heading %f xR %f yR %f", heading, xPos, yPos);
    putText(dst, str,Point(5,90), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    applyText(statusText, Point(5, 90), dst);
    //draw crosshairs
    line(dst, Point( IMAGE_WIDTH/2, 0), Point(IMAGE_WIDTH / 2, IMAGE_HEIGHT), Scalar(0, 255, 255), 1, 8, 0);
    line(dst, Point( 0, IMAGE_HEIGHT/2), Point(IMAGE_WIDTH, IMAGE_HEIGHT/2), Scalar(0, 255, 255), 1, 8, 0);
    /// Show Images
    if (displayMode == WindowMode::FINAL) {
        WindowMode::print(displayMode, dst);
        Window::print("Ratchet Rockers 1706", dst, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, dst);
    }
}

const double cameraFOV = 117.5; // degrees
const double ballWidth = 0.6096; // meters

Point2d lastBallPosition = {0, 0};
deque<Point2d> lastBallPositions;
auto lastFrameStart = std::chrono::high_resolution_clock::now();
SolutionLog ballPositions;
int ballFrameCount = 0;
void ballDetection(ThreadData &data)
{
    Mat img = data.image;
    int IMAGE_WIDTH = img.cols, IMAGE_HEIGHT = img.rows;
    if (!ballPositions.isOpen()) {
        sprintf(str, "%s/balltrack.csv", dirname);
        ballPositions.open(str, {"frame", "time", "image", "pos_px_x", "pos_px_y", "distance", "rotation"});
    }
    assert(inputType == COLOR); // Ball can only be detected on color image
    auto timeNow = std::chrono::high_resolution_clock::now();
    double timeSinceLastFrame = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(timeNow-lastFrameStart).count()) / 1000000000.0;
    Mat dst = img.clone();
    if (displayMode == WindowMode::RAW) {
        WindowMode::print(displayMode, dst);
        Window::print("Ratchet Rockers 1706", dst, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, dst);
    }
    cvtColor(img,img, CV_BGR2RGB);
    cvtColor(img, img, CV_BGR2HSV);
    // Threshold image to
    inRange(img, Scalar(ballHueMin, ballSatMin, ballValMin), Scalar(ballHueMax, ballSatMax, ballValMax), img);
    if (displayMode == WindowMode::THRESHOLD) {
        Mat thresh = img.clone();
        cvtColor(thresh, thresh, CV_GRAY2RGB);
        WindowMode::print(displayMode, thresh);
        int curThreshVal;
        switch (currentThreshold) {
        case Thresh::HUE_MIN:
            curThreshVal = ballHueMin;
            break;
        case Thresh::HUE_MAX:
            curThreshVal = ballHueMax;
            break;
        case Thresh::SAT_MIN:
            curThreshVal = ballSatMin;
            break;
        case Thresh::SAT_MAX:
            curThreshVal = ballSatMax;
            break;
        case Thresh::VAL_MIN:
            curThreshVal = ballValMin;
            break;
        case Thresh::VAL_MAX:
            curThreshVal = ballValMax;
            break;
        default:
            abort();
        }
        sprintf(str, "%s: %d", Thresh::str(currentThreshold).c_str(), curThreshVal);
        putText(thresh, str, Point(5, 30), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,255), 1, 8, false);
        Window::print("Ratchet Rockers 1706", thresh, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, thresh);
    }
    // Get rid of remaining noise
    morphologyEx(img, img, MORPH_OPEN, kernel, Point(-1, -1), dilations);
    if (displayMode == WindowMode::DILATE) {
        Mat dilate = img.clone();
        cvtColor(dilate, dilate, CV_GRAY2RGB);
        WindowMode::print(displayMode, dilate);
        sprintf(str, "Dilations: %d", dilations);
        putText(dilate, str, Point(5, 30), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,255), 1, 8, false);
        Window::print("Ratchet Rockers 1706", dilate, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, dilate);
    }
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
    if (displayMode == WindowMode::CONTOURS || displayMode == WindowMode::APPROXPOLY) {
        Mat contoursImg = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);
        cvtColor(contoursImg, contoursImg, CV_GRAY2RGB);
        for ( unsigned int i = 0; i < contours.size(); i++) {
            vector<Point> contour = contours[i];
            if (contourArea(contour) < contourMinArea) {
                continue;
            }
            if (displayMode == WindowMode::CONTOURS) {
                drawContours(contoursImg, contours, i, Scalar(255, 255, 0));
            }
            if (displayMode == WindowMode::APPROXPOLY) {
                Point2f ballCenterFlat;
                float radius;
                minEnclosingCircle(contour, ballCenterFlat, radius);
                circle(contoursImg, ballCenterFlat, (int)radius, Scalar(0, 0, 255), 2, 8, 0 );
            }
        }
        if (displayMode == WindowMode::CONTOURS) {
            sprintf(str, "Area: %d", contourMinArea);
        } else if (displayMode == WindowMode::APPROXPOLY) {
            sprintf(str, "Accuracy: %d", accuracy);
        }
        WindowMode::print(displayMode, contoursImg);
        putText(contoursImg, str, Point(5, 30), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,255), 1, 8, false);
        Window::print("Ratchet Rockers 1706", contoursImg, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, contoursImg);
    }
    vector<vector<Point> > succeededContours = getSuccessfulContours(contours, ballTests);
    vector<vector<Point> > largestContour(static_cast<unsigned int>(1));
    for (vector<Point> &contour : succeededContours) {
        if (largestContour.size() > 0 && largestContour[0].size() > 0) {
            if (contourArea(contour) > contourArea(largestContour[0])) {
                largestContour[0] = contour;
            }
        } else {
            largestContour[0] = contour;
        }
    }
    if (largestContour[0].size() == 0) {
        largestContour.pop_back();
    }
    if (displayMode == WindowMode::PASS) {
        Mat pass = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);
        cvtColor(pass, pass, CV_GRAY2RGB);
        drawContours(pass, largestContour, 0, Scalar(255, 255, 0));
        WindowMode::print(displayMode, pass);
        Window::print("Ratchet Rockers 1706", pass, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, pass);
    }
    double angleToBall = 0;
    double ballVelocity = 0;
    double distanceToBall = 0;
    double ballHeading = 0;
    for (vector<Point> &contour : largestContour) {
        vector<Point> polygon;
        approxPolyDP( contour, polygon, accuracy, true );
        Point2f ballCenterFlat;
        float radius;
        minEnclosingCircle(contour, ballCenterFlat, radius);
        circle( dst, ballCenterFlat, (int)radius, Scalar(0, 0, 255), 2, 8, 0 );
        double diameter = radius * 2.0;
        double ballAngle = (cameraFOV * diameter) / IMAGE_WIDTH;
        distanceToBall = (1.0 / tan((ballAngle / 2.0) * (CV_PI / 180))) * (ballWidth / 2.0);
        line(dst, Point(ballCenterFlat.x, 0), Point(ballCenterFlat.x, IMAGE_HEIGHT), Scalar(0, 255, 50));
        line(dst, Point(0, ballCenterFlat.y), Point(IMAGE_WIDTH, ballCenterFlat.y), Scalar(0, 255, 50));
        ballCenterFlat.x = (ballCenterFlat.x - IMAGE_WIDTH / 2); // rebase origin to center
        ballCenterFlat.y = -(ballCenterFlat.y - IMAGE_HEIGHT / 2);
        double ballPosXreal = (ballWidth * ballCenterFlat.x) / diameter;
        double ballPosYreal = sqrt(square(distanceToBall) - square(ballPosXreal));
        Point3d ballCenter = Point3d(ballPosXreal, ballPosYreal, distanceToBall);
        Point2d centerXY = Point2d(ballPosXreal, ballPosYreal);
        angleToBall = acos(centerXY.y / distanceToBall) * (180 / CV_PI);
        Point2d change = lastBallPosition - centerXY;
        double movedDistance = sqrt(square(change.x) + square(change.y)); // real, meters
        ballVelocity = movedDistance / timeSinceLastFrame; // meters per second
        ballHeading = acos(change.x / movedDistance) * (180 / CV_PI);
        Point pos = contour[0];
        sprintf(str, "Dia:%.2fpx", diameter);
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        pos.y += 10;
        sprintf(str, "BallAng:%.2fo", ballAngle);
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        pos.y += 10;
        sprintf(str, "Dist:%.2fm", distanceToBall);
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        // Notes on position ball
        // The plane on which XY resides is a top-view basically
        // X is the distance over from the center of the camera
        // Y is the distance from the camera to the ball
        // See connor's engineering notebook for more, page 23
        sprintf(str, "Center:%s", xyz(ballCenter).c_str());
        pos.y += 10;
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        sprintf(str, "Change:%s", xy(change).c_str());
        pos.y += 10;
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        sprintf(str, "Moved:%.2fm Angle:%.2f Velocity:%.2fm/s", movedDistance, angleToBall, ballVelocity);
        pos.y += 10;
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        sprintf(str, "Heading:%.2f", ballHeading);
        pos.y += 10;
        putText(dst, str, pos, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
        lastBallPosition = centerXY;
        if (lastBallPositions.size() > 10) {
            lastBallPositions.pop_front();
        }
        lastBallPositions.push_back(centerXY);
        // store five points here
        // calculate median of first five for first point
        // calculate median of last five for second point
        ballPositions.log("pos_px_x", ballCenterFlat.x).log("pos_px_y", ballCenterFlat.y);
        ballPositions.log("distance", distanceToBall).log("rotation", ballHeading);
    }
    line(dst, Point(IMAGE_WIDTH / 2, 0), Point(IMAGE_WIDTH / 2, IMAGE_HEIGHT), Scalar(0, 255, 255));
    line(dst, Point(0, IMAGE_HEIGHT / 2), Point(IMAGE_WIDTH, IMAGE_HEIGHT / 2), Scalar(0, 255, 255));
    if (displayMode == WindowMode::FINAL) {
        WindowMode::print(displayMode, dst);
        Window::print("Ratchet Rockers 1706", dst, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, dst);
    }
    if (doUdp) {
        QByteArray datagram = "balltrack "
                + QByteArray::number(distanceToBall) + " "
                + QByteArray::number(angleToBall) + " "
                + QByteArray::number(ballVelocity);

        udpSocket.writeDatagram(datagram.data(), datagram.size(), udpRecipient, 8888);
    }
    if (ballPositions.isOpen()) {
        double timeSinceStart = std::chrono::duration_cast<std::chrono::duration<double> >(timeNow-startTime).count();
        ballPositions.log("frame", ballFrameCount).log("time", timeSinceStart).log("image", imageWriteIndex).flush();
    } else {
        cerr << "Unable to write solutions log" << endl;
    }
    lastFrameStart = timeNow;
    ballFrameCount++;
}
