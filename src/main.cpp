#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "util.hpp"

#define FOV_Y 79//degrees
#define FOV_X 111.426 //degrees
#define STATIC_TARGET_HEIGHT  32.25 //in

// OpenCV Namespace
using namespace cv;
using namespace std;

// for sprintf (grr)
char str[255];

// keys
const char KEY_QUIT = 27;
const char KEY_SAVE = 's';
const char KEY_SPEED = ' ';

// config
const Mode mode = CAMERA;
const int cameraId = 1;
const CaptureMode inputType = IR;
const string videoPath = "Y400cmX646cm.avi";
const bool displayImage = true;

// Values for threshold IR
const int gray_min = 245;
const int gray_max = 255;

// Values for threshold RGB
const int hue_min = 35;
const int hue_max = 90;
const int saturation_min = 10;
const int saturation_max = 255;
const int value_min = 140;
const int value_max = 255;

// for approxpolydp
const int accuracy = 3; //maximum distance between the original curve and its approximation
const int contourMinArea = 50;

const float calibrationRange = 2.724; // meters
const float calibrationPixels = 10; // pixels

const int kern_mat[] = {1,0,1,
                        0,1,0,
                        1,0,1};
const Mat kernel = getStructuringElement(*kern_mat, Size(3,3), Point(-1,-1));
const Size winSize = Size( 5, 5 );
const Size zeroZone = Size( -1, -1 );
const TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );


int IMAGE_WIDTH = 0;
int IMAGE_HEIGHT = 0;

template<class ArrayOfPoints>
void T2B_L2R(ArrayOfPoints pt)
{
    int temp_x;
    int temp_y;

    int i, swapped;

    do {
        swapped = 0;
        for (i = 1; i < 4; i++)
        {
            if (pt[i-1].y > pt[i].y)
            {
                temp_x = pt[i-1].x;
                temp_y = pt[i-1].y;
                pt[i-1].x = pt[i].x;
                pt[i-1].y = pt[i].y;
                pt[i].x = temp_x;
                pt[i].y = temp_y;
                swapped = 1;
            }
        }
    }
    while (swapped == 1);

    /// Make sure top two points are left to right
    if (pt[0].x > pt[1].x)
    {
        temp_x = pt[0].x;
        temp_y = pt[0].y;
        pt[0].x = pt[1].x;
        pt[0].y = pt[1].y;
        pt[1].x = temp_x;
        pt[1].y = temp_y;
    }

    /// Make sure bottom two points are left to right
    if (pt[2].x > pt[3].x)
    {
        temp_x = pt[2].x;
        temp_y = pt[2].y;
        pt[2].x = pt[3].x;
        pt[2].y = pt[3].y;
        pt[3].x = temp_x;
        pt[3].y = temp_y;
    }
}

void applyText(vector<string> &text, Point startPos, Mat &img)
{
    int x = startPos.x;
    int y = startPos.y;
    std::vector<string>::const_iterator iterator;
    for (iterator = text.begin(); iterator != text.end(); ++iterator) {
        putText(img, (*iterator).c_str(), Point(x, y), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
        y += 20;
    }
}

void targetDetection(Mat img, int id);

int main()
{
    VideoCapture camera;
    if (mode == CAMERA) {
        camera = VideoCapture(cameraId);
        if (!camera.isOpened()) {
            cerr << "Failed to open camera device id:" << cameraId << endl;
            return -1;
        }
        camera.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        camera.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    } else if (mode == VIDEO) {
        camera = VideoCapture(videoPath);
        if (!camera.isOpened()) {
            cerr << "Failed to open video path:" << videoPath << endl;
            return -1;
        }
    }
    IMAGE_HEIGHT =  camera.get(CV_CAP_PROP_FRAME_HEIGHT);
    IMAGE_WIDTH = camera.get(CV_CAP_PROP_FRAME_WIDTH);
    int currentSave = 0;

    Mat img, inframe;

    if (mode == IMAGE) {
        inframe = imread("raw_img_0.png");
    }

    while ( 1 )
    {
        //reset variables
        int Plane_Distance = 0;
        int Image_Heigh_in = 0;
        int Real_Distance = 0;
        int In_Screen_Angle = 0;
        int failedArea = 0;
        int failedHierarchy = 0;
        int failedSides = 0;
        int failedConvex = 0;
        int failedSquare = 0;
        int failedVLarge = 0;
        int success = 0;
        // Start timing a frame (FPS will be a measurement of the time it takes to process all the code for each frame)
        auto start = std::chrono::high_resolution_clock::now();

        if (mode == CAMERA || mode == VIDEO) { // Replaced #if with braced conditional. Modern compiler should have no performance differences.
            // Grab a frame and contain it in the cv::Mat img
            camera >> img;
            if (&img == NULL) {
                cerr << "Video stream ended abruptly." << endl;
                return -1;
            }
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
                camera >> img;
            }
            break;
        case 'p':
            waitKey();
            break;
        }
        targetDetection(img, 0);
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

void targetDetection(Mat img, int)
{
    // Store the original image img to the Mat dst
    Mat dst = img.clone();

    // Convert image from input to threshold method
    if (inputType == IR) {
        cvtColor(img, img, CV_BGR2GRAY);
    } else if (inputType == COLOR) {
        cvtColor(img, img, CV_BGR2HSV);
    }
    if (displayImage) {
        Mat input = dst.clone();
        sprintf(str, "Input Mode = %s", inputType == IR ? "IR" : "Color");
        putText(input, str, Point(5,5), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
        imshow("Target Input", input);
    }

    // "Threshold" image to pixels in the ranges
    if (inputType == IR) {
        threshold(img, img, gray_min, gray_max, CV_THRESH_BINARY);
    } else if (inputType == COLOR) {
        inRange(img, Scalar(hue_min, saturation_min, value_min), Scalar(hue_max, saturation_max, value_max), img);
    }

    // Get rid of remaining noise
    dilate(img, img, kernel);
    erode(img, img, kernel, Point(-1, -1), 2);

    if (displayImage) {
        imshow("Target Dialate", img);
    }

    // Declare containers for contours and contour heirarchy
    vector<vector<Point> > contours;
    vector<Point2f> corners;
    vector<vector<Point> > Static_Target;
    vector<vector<Point> > Dynamic_Target;
    vector<Vec4i> hierarchy;


    findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    double ratio;
    double lengthTop;
    double lengthBottom;
    double lengthRight;
    double lengthLeft;
    vector<string> statusText;
    int totalContours = contours.size();
    int failedArea = 0;
    int failedHierarchy = 0;
    int failedSides = 0;
    int failedConvex = 0;
    int failedSquare = 0;
    int failedVLarge = 0;
    int success = 0;
    double Image_Heigh_in = 0;
    double Plane_Distance = 0;
    float Tan_FOV_Y_Half = 1.46;

    // Create a for loop to go through each contour (i) one at a time
    for( unsigned int i = 0; i < contours.size(); i++ )
    {
        vector<Point> contour = contours[i];
        Vec4i contourHierarchy = hierarchy[i];
        // Very small contours (noise)
        if (contourArea(contour) < contourMinArea) {
            failedArea++;
            continue;
        }
        vector<Point> polygon;
        approxPolyDP( contour, polygon, accuracy, true );
        vector<vector<Point>> contoursDrawWrapper {polygon};
        Rect boundRect = boundingRect(polygon);
        rectangle( dst, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0), 2, 8, 0 );

        // ratio helps determine orientation of rectangle (vertical / horizontal)
        ratio = static_cast<double>(boundRect.width) / static_cast<double>(boundRect.height);
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
        lengthTop = distance(localCorners[0], localCorners[1]);
        lengthBottom = distance(localCorners[2], localCorners[3]);
        lengthLeft = distance(localCorners[0], localCorners[2]);
        lengthRight = distance(localCorners[1], localCorners[3]);
        success++;
        int centerX = boundRect.x + boundRect.width / 2;
        int centerY = boundRect.y + boundRect.height / 2;
        Point2i center = {centerX, centerY};

        if (boundRect.height > boundRect.width * 2) // static target
        {
            double refinedHeight = distance(localCorners[0], localCorners[2]);
            double flatHeight = localCorners[2].y - localCorners[0].y;
            double Center_Static_X = (boundRect.x + (boundRect.width / 2)) - (IMAGE_WIDTH/2);
            Image_Heigh_in = (IMAGE_HEIGHT * STATIC_TARGET_HEIGHT) / boundRect.height;
            Plane_Distance = (Image_Heigh_in) / Tan_FOV_Y_Half;
            double In_Screen_Angle = (FOV_X / IMAGE_WIDTH) * Center_Static_X;
            double Real_Distance = Plane_Distance / (cos(In_Screen_Angle * CV_PI / 180));

            assert(Real_Distance >= 0);

            sprintf(str, "BRH:%d", boundRect.height);
            putText(dst, str, center, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            sprintf(str, "RFH:%.2f", refinedHeight);
            putText(dst, str, center + Point2i(0, 15), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            sprintf(str, "PLH:%.2f", flatHeight);
            putText(dst, str, center + Point2i(0, 30), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            sprintf(str, "PLD:%.2f", Plane_Distance);
            putText(dst, str, center + Point2i(0, 45), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            sprintf(str, "RLD:%.2f", Real_Distance);
            putText(dst, str, center + Point2i(0, 60), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            circle(dst, localCorners[0], 5, Scalar(0, 255, 255), 2, 8, 0);
            circle(dst, localCorners[2], 5, Scalar(100, 255, 200), 2, 8, 0);
            //contour is a tall and skinny one
            //save off as static target
            Static_Target.push_back(contours[i]);
        }
        else
        {
            //contour is the short and wide, dynamic target
            //save off as dynamic target
            Dynamic_Target.push_back(contours[i]);
            drawContours(dst, contoursDrawWrapper,0, Scalar(255, 0, 0), 3, 8, hierarchy, 0, Point() );
        }
    }
    cout << "Total: " << totalContours << " | Failures Area: " << failedArea << " Hierarchy: " << failedHierarchy <<
            " Sides: " << failedSides << " Convex: " << failedConvex << " Square: " << failedSquare << " VeryLarge: " << failedVLarge << " | Success: " << success << endl;

    /// calculate center
    vector<Moments> Moment_Center_Static(Static_Target.size() );
    for( unsigned int i = 0; i < Static_Target.size(); i++ )
    {
        Moment_Center_Static[i] = moments( Static_Target[i], false );
    }

    ///  Get the mass centers:
    vector<Point2f> Mass_Center_Static( Static_Target.size() );
    for( unsigned int i = 0; i < Static_Target.size(); i++ )
    {
        Mass_Center_Static[i] = Point2f( Moment_Center_Static[i].m10/Moment_Center_Static[i].m00 , Moment_Center_Static[i].m01/Moment_Center_Static[i].m00 );
    }

    /// calculate center
    vector<Moments> Moment_Center_Dynamic(Dynamic_Target.size() );
    for( unsigned int i = 0; i < Dynamic_Target.size(); i++ )
    {
        Moment_Center_Dynamic[i] = moments( Dynamic_Target[i], false );
    }

    ///  Get the mass centers:
    vector<Point2f> Mass_Center_Dynamic( Dynamic_Target.size() );
    for( unsigned int i = 0; i < Dynamic_Target.size(); i++ )
    {
        Mass_Center_Dynamic[i] = Point2f( Moment_Center_Dynamic[i].m10/Moment_Center_Dynamic[i].m00 , Moment_Center_Dynamic[i].m01/Moment_Center_Dynamic[i].m00 );
    }
    TargetCase targetCase = NONE;
    if(Mass_Center_Static.size() > 0 && Mass_Center_Dynamic.size() > 0
            && Mass_Center_Static[0].x > Mass_Center_Dynamic[0].x) {
        //case left
        if (Static_Target.size() + Dynamic_Target.size() == 2) {
            targetCase = LEFT;
        }
    } else {
        //case right
        if (Static_Target.size() + Dynamic_Target.size() == 2) {
            targetCase = RIGHT;
        }
    }
    if (Static_Target.size() + Dynamic_Target.size() == 4) {
        targetCase = ALL;
    }

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
    sprintf(str, "Targets S:%ld D:%ld", Static_Target.size(), Dynamic_Target.size());
    putText(dst, str,Point(5,60), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    sprintf(str, "Image Height %dpx %.2fin", IMAGE_HEIGHT, Image_Heigh_in);
    putText(dst, str,Point(5,75), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    applyText(statusText, Point(5, 90), dst);
    //draw crosshairs
    line(dst, Point( IMAGE_WIDTH/2, 0), Point(IMAGE_WIDTH / 2, IMAGE_HEIGHT), Scalar(0, 255, 255), 1, 8, 0);
    line(dst, Point( 0, IMAGE_HEIGHT/2), Point(IMAGE_WIDTH, IMAGE_HEIGHT/2), Scalar(0, 255, 255), 1, 8, 0);
    /// Show Images
    if (displayImage) {
        imshow("Target Detection", dst);
    }
}
