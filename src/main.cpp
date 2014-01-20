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

#define FOV_Y 80//degrees
#define FOV_X 117.5 //degrees
#define IMAGE_HEIGHT 480
#define IMAGE_WIDTH 640
#define STATIC_TARGET_HEIGHT  32 //in

// OpenCV Namespace
using namespace cv;
using namespace std;

//constants
const int ESC = 27;
const int contourMinArea = 50;
const Mode mode = IMAGE;
const int cameraId = 1;
const CaptureMode inputType = IR;
const string videoPath = "Y400cmX646cm.avi";

void T2B_L2R(CvPoint* pt)
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

int main()
{
    // Values for threshold IR
    const int gray_min = 245;
    const int gray_max = 255;

    // Values for threshold RGB
    const int hue_min = 55;
    const int saturation_min = 10;
    const int value_min = 190;
    const int hue_max = 250;
    const int saturation_max = 155;
    const int value_max = 255;
    float FOV_X_Divided_Image_Width = FOV_X / IMAGE_WIDTH;
    float Tan_FOV_Y_Half = 0.8390;
    const float FOV_Y_DIV_HEIGHT = FOV_Y / IMAGE_HEIGHT;
    float Image_Heigh_in;
    float Plane_Distance;
    float Real_Distance;

    int kern_mat[] = {1,0,1,
                      0,1,0,
                      1,0,1};

    Mat kernel = getStructuringElement(*kern_mat, Size(3,3), Point(-1,-1));

    // for approxpolydp
    const int accuracy = 9; //maximum distance between the original curve and its approximation

    const float calibrationRange = 2.724; // meters
    const float calibrationPixels = 10; // pixels

    // create a storage for the corners for ration test
    CvPoint pt[4];

    // Define character to store a string of up to 50 characters, to be printed on the image
    char str[50];

    // Set the neeed parameters to find the refined corners
    Size winSize = Size( 5, 5 );
    Size zeroZone = Size( -1, -1 );
    TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

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
    int currentSave = 0;

    // Create Mat to store images
    Mat img, dst, thresh, inframe;

    // Create Windows
    namedWindow("Final", CV_WINDOW_NORMAL);

    if (mode == IMAGE) {
        inframe = imread("raw_img_0.png");
    }

    while ( 1 )
    {
        //reset variables
        Plane_Distance = 0;
        Image_Heigh_in = 0;
        Real_Distance = 0;
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
        case ESC:
            return 0;
            break;
        case 's':
            sprintf(str, "raw_img_%d.png", currentSave);
            imwrite(str, img);
            currentSave++;
            break;
        case ' ':
            for (int gi = 0; gi < 20; gi++) {
                camera >> img;
            }
            break;
        }

        // Store the original image img to the Mat dst
        img.copyTo(dst);

        // Convert image from input to threshold method
        if (inputType == IR) {
            cvtColor(img, img, CV_BGR2GRAY);
        } else if (inputType == COLOR) {
            cvtColor(img, img, CV_BGR2HSV);
        }
        Mat input = img.clone();
        sprintf(str, "Input Mode = %s", inputType == IR ? "IR" : "Color");
        putText(input, str, Point(5,5), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
        imshow("Input", input);

        // "Threshold" image to pixels in the ranges
        if (inputType == IR) {
            threshold(img, img, gray_min, gray_max, CV_THRESH_BINARY);
        } else if (inputType == COLOR) {
            inRange(img, Scalar(hue_min, saturation_min, value_min), Scalar(hue_max, saturation_max, value_max), img);
        }
        imshow("Threshold", img);

        // Get rid of remaining noise
        dilate(img, img, kernel);
        erode(img, img, kernel, Point(-1, -1), 2);
        dilate(img, img, kernel);
        imshow("Dilate", img);


        // Declare containers for contours and contour heirarchy
        vector<vector<Point> > contours;
        vector<Point2f> corners;
        vector<vector<Point> > Static_Target;
        vector<vector<Point> > Dynamic_Target;
        vector<Vec4i> hierarchy;

        findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Declare container for approximated polygons
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );

        double ratio;
        double lengthTop;
        double lengthBottom;
        double lengthRight;
        double lengthLeft;
        vector<string> statusText;
        int totalContours = contours.size();

        // Create a for loop to go through each contour (i) one at a time
        for( unsigned int i = 0; i < contours.size(); i++ )
        {
            // Very small contours (noise)
            if (contourArea(contours[i]) < contourMinArea) {
                failedArea++;
                continue;
            }

//            // Contours that have an interior contour
//            if (hierarchy[i][0] <= 0) {
//                failedHierarchy++;
//                continue;
//            }

//            // Polygon does not have four sides
//            if (contours_poly[i].size() != 4) {
//                failedSides++;
//                continue;
//            }

            approxPolyDP( contours[i], contours_poly[i], accuracy, true );

//            // Non-regular polygons
//            if (!isContourConvex(contours_poly[i])) {
//                failedConvex++;
//                continue;
//            }

            boundRect[i] = boundingRect(contours_poly[i]);

            // ratio helps determine orientation of rectangle (vertical / horizontal)
            ratio = static_cast<double>(boundRect[i].width) / static_cast<double>(boundRect[i].height);
            if (isAlmostSquare(ratio)) {
                failedSquare++;
                continue;
            } else if (isExtraLong(ratio)) {
                failedVLarge++;
                continue;
            }

            rectangle( dst, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );

            vector<Point2f> localCorners;
            for (int k = 0; k < 4; k++)
            {
                pt[k] = contours_poly[i][k];
            }

            // organize corners
            T2B_L2R(pt);

            //populate localCorners with pt
            for (int k = 0; k < 4; k++)
            {
                localCorners.push_back(pt[k]);
            }

            // Calculate the refined corner locations
            cornerSubPix(img, localCorners, winSize, zeroZone, criteria);
            corners.insert(corners.end(), localCorners.begin(), localCorners.end());

            // test aspect ratio
            lengthTop = distance(pt[0], pt[1]);
            lengthBottom = distance(pt[2], pt[3]);
            lengthLeft = distance(pt[0], pt[2]);
            lengthRight = distance(pt[1], pt[3]);
            success++;
            int centerX = boundRect[i].x + boundRect[i].width / 2;
            int centerY = boundRect[i].y + boundRect[i].height / 2;

            if (boundRect[i].height > boundRect[i].width)
            {
                double Center_Static_X = -((IMAGE_WIDTH/2) - ( boundRect[i].x + (boundRect[i].width / 2))) ;
//                double Center_Static_Y = -((boundRect[i].y + boundRect[i].width) - (IMAGE_HEIGHT/2));
                double Center_Static_Y = -(centerY - (IMAGE_HEIGHT/2));
                Image_Heigh_in = (IMAGE_HEIGHT * STATIC_TARGET_HEIGHT) / boundRect[i].height;
                Plane_Distance = (Image_Heigh_in * 0.5) / Tan_FOV_Y_Half;
                double In_Screen_Angle_X = (FOV_X / IMAGE_WIDTH) * Center_Static_X;
                double In_Screen_Angle_Y = (FOV_Y / IMAGE_HEIGHT) * Center_Static_Y;
                double Real_Distance_X = Plane_Distance / (cos(In_Screen_Angle_X * CV_PI / 180));
                Point isa(In_Screen_Angle_X, In_Screen_Angle_Y);
                Real_Distance = Real_Distance_X / (cos(In_Screen_Angle_Y * CV_PI / 180));
//                Real_Distance = sqrt(square(Real_Distance_X) + 256);
                assert(Real_Distance >= 0);

                sprintf(str, "IHi:%f PD:%f ISA:%s RD:%f", Image_Heigh_in, Plane_Distance, xyz(isa).c_str(), Real_Distance);
                statusText.push_back(str);
            }

            if (ratio < 1) //subject to change
            {
                //contour is a tall and skinny one
                //save off as static target
                Static_Target.push_back(contours[i]);
                drawContours(dst, contours_poly,i, Scalar(0,0,255), 3, 8, hierarchy, 0, Point() );
                sprintf(str, "LC0:%s LC1:%s", xyz(localCorners[0]).c_str(), xyz(localCorners[1]).c_str());
                statusText.push_back(str);
                int lengthStaticTop = distance(localCorners[0], localCorners[1]);
                float distanceToTarget = (calibrationRange / lengthStaticTop) * calibrationPixels;
                sprintf(str, "R:%f L:%dpx D:%fm", ratio, lengthStaticTop, distanceToTarget);
                statusText.push_back(str);
//                sprintf(str, "H:%f L:%f", distance(localCorners[0], localCorners[2]), distance(localCorners[0], localCorners[1]));
                sprintf(str, "H:%d L:%d", boundRect[i].height, boundRect[i].width);
                statusText.push_back(str);
                sprintf(str, "BRCX:%d BRCY:%d", boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height);
                statusText.push_back(str);
            }
            else
            {
                //contour is the short and wide, dynamic target
                //save off as dynamic target
                Dynamic_Target.push_back(contours[i]);
                drawContours(dst, contours_poly,i, Scalar(255, 0, 0), 3, 8, hierarchy, 0, Point() );
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
                && Mass_Center_Static[0].x > Mass_Center_Dynamic[0].x)
        {
            //case left
            if (Static_Target.size() + Dynamic_Target.size() == 2) {
                targetCase = LEFT;
            }
        }

        else
        {
            //case right
            if (Static_Target.size() + Dynamic_Target.size() == 2) {
                targetCase = RIGHT;
            }
        }
        if (Static_Target.size() + Dynamic_Target.size() == 4) {
            targetCase = ALL;
        }

        /// Stop timing and calculate FPS and Average FPS
        auto finish = std::chrono::high_resolution_clock::now();
        double seconds = ((double) std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count()) / 1000000000.0;
        sprintf(str, "FPS %d, per frame: %fs", (int) (1 / seconds), seconds);
        putText(dst, str,Point(5,15), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
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
        applyText(statusText, Point(5, 90), dst);

        //draw crosshairs
        line(dst, Point( IMAGE_WIDTH/2, 0), Point(IMAGE_WIDTH / 2, IMAGE_HEIGHT), Scalar(0, 255, 255), 1, 8, 0);
        line(dst, Point( 0, IMAGE_HEIGHT/2), Point(IMAGE_WIDTH, IMAGE_HEIGHT/2), Scalar(0, 255, 255), 1, 8, 0);

        // Show Images
        imshow("Final", dst);
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
