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

/// OpenCV Namespace
using namespace cv;
using namespace std;

#define Minimum_Area 500
#define ESC 27
#define USE_CAMERA 1
#define CAMERA 0
enum TargetCase {
    NONE = 0,
    ALL = 1,
    RIGHT = 2,
    LEFT = 3
};


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
    // Values for inrange
    int hue_min = 35;
    int saturation_min = 0;
    int value_min = 155;
    int hue_max = 90;
    int saturation_max = 255;
    int value_max = 255;

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
    if (USE_CAMERA) {
        camera = VideoCapture(CAMERA);
        if (!camera.isOpened()) {
            cerr << "Failed to open camera device id:" << CAMERA << endl;
            return -1;
        }
//        cout << "CV_CAP_PROP_FOURCC=" << camera.get(CV_CAP_PROP_FOURCC) << endl;
        camera.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
        camera.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    }

    Mat img, dst, thresh, inframe;
    // Create Windows
    namedWindow("Raw", CV_WINDOW_AUTOSIZE);
    namedWindow("HSV", CV_WINDOW_AUTOSIZE);
    namedWindow("Final", CV_WINDOW_NORMAL);

    if (!USE_CAMERA) {
        inframe = imread("raw_img.jpg");
    }

    while ( 1 )
    {
        // Start timing a frame (FPS will be a measurement of the time it takes to process all the code for each frame)
        auto start = std::chrono::high_resolution_clock::now();

        if (USE_CAMERA) { // Replaced #if with braced conditional. Modern compiler should have no performance differences.
            // Grab a frame and contain it in the cv::Mat img
            camera >> img;
        } else {
            img = inframe.clone();
        }

        //Break out of loop if esc is pressed
        switch (char key = waitKey(10)) {
        case ESC:
            return 0;
            break;
        case 's':
            imwrite("raw_img.jpg", img);
        }

        // Store the original image img to the Mat dst
        img.copyTo(dst);

        // Convert img from BGR(Blue,Green,Red) to HSV(Hue,Saturation,Value)
        cvtColor(img,img,CV_BGR2HSV);

        // "Threshold" image to pixels in the ranges
        inRange(img,Scalar(hue_min, saturation_min, value_min), Scalar(hue_max, saturation_max, value_max),img);

        // Save img to thresh Mat to show to the user
        img.copyTo(thresh);

        // Get rid of remaining noise
        erode(img, img, 0.0);
        dilate(img, img, 0.0);


        // Declare containers for contours and contour heirarchy
        vector<vector<Point> > contours;
        vector<Point2f> corners;
        vector<vector<Point> > Static_Target;
        vector<vector<Point> > Dynamic_Target;
        vector<Vec4i> hierarchy;


        findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );

        // Declare container for approximated polygons
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );

        double ratio;
        double lengthTop;
        double lengthBottom;
        double lengthRight;
        double lengthLeft;
        vector<string> statusText;

        // Create a for loop to go through each contour (i) one at a time
        for( unsigned int i = 0; i < contours.size(); i++ )
        {
            // If contour has an area greater than 500 pixels
            if(contourArea(contours[i])>Minimum_Area)
            {
                // Approximate a Polygon and save to contours_poly
                approxPolyDP( contours[i], contours_poly[i], accuracy, true );

                // If 4 sided
                if(contours_poly[i].size()== 4 && isContourConvex(contours_poly[i]))
                {

                    boundRect[i] = boundingRect(contours_poly[i]);

                    rectangle( dst, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );
                    vector<Point2f> localCorners;
                    for (int k = 0; k < 4; k++)
                    {
                        pt[k] = contours_poly[i][k];
                    }



                    // organize corners
                    T2B_L2R(pt);
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

                    // ratio helps determine orientation of rectangle (vertical / horizontal)
                    ratio = static_cast<double>(boundRect[i].width) / static_cast<double>(boundRect[i].height);
                    if (isAlmostSquare(ratio)) {
                        cout << "Ignoring ratio " << ratio << endl;
                        continue; // go to next contour
                    } else {
                        cout << "Ratio of " << ratio << " is a target" << endl;
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
                        sprintf(str, "H:%f L:%f", distance(localCorners[0], localCorners[2]), distance(localCorners[0], localCorners[1]));
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

            }
        }

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
        /// Show Images
        imshow("Raw", img);
        imshow("HSV", thresh);
        imshow("Final", dst);
        if (!USE_CAMERA) {
            char k = waitKey(); // pause
            if (k >= '0' && k <= '9') {
                stringstream filename;
                filename << "raw_img_" << k << ".jpg";
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
