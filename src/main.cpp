#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include "timer.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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

float length(float x1, float y1, float x2, float y2)
{
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

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
int main()
{
    // Values for inrange
    int hue_min = 60;
    int saturation_min = 30;
    int value_min = 155;
    int hue_max = 255;
    int saturation_max = 255;
    int value_max = 255;

    // for approxpolydp
    const int accuracy = 9; //maximum distance between the original curve and its approximation

    // create a storage for the corners for ration test
    CvPoint pt[4];
    CvPoint2D32f cornersubpix[4];

    // Declare Timing and create variables to store frame times
    DECLARE_TIMING(FPS);
    double time_now;
    double time_ave;

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
    }

    // Initialize a font, used for printed text onto an image
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, 0.75, 0, 1, CV_AA);

    Mat img, dst, thresh, inframe;
    // Create Windows
    namedWindow("Raw", CV_WINDOW_AUTOSIZE);
    namedWindow("HSV", CV_WINDOW_AUTOSIZE);
    namedWindow("Final", CV_WINDOW_AUTOSIZE);

    if (!USE_CAMERA) {
        inframe = imread("raw_img.jpg");
    }

    while ( 1 )
    {
        // Start timing a frame (FPS will be a measurement of the time it takes to process all the code for each frame)
        START_TIMING(FPS);

        //Break out of loop if esc is pressed
        switch (char key = waitKey(10)) {
        case ESC:
            return 0;
            break;
        case 's':
            imwrite("raw_img.jpg", img);
        }

        if (USE_CAMERA) { // Replaced #if with braced conditional. Modern compiler should have no performance differences.
            // Grab a frame and contain it in the cv::Mat img
            camera >> img;
        } else {
            inframe.copyTo(img);
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
        erode(img, img, NULL);
        dilate(img, img, NULL);


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

        double Ratio;
        double Ratio_Top;
        double Ratio_Bottom;
        double Ratio_Right;
        double Ratio_Left;

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

                    for ( int j = 0; j < 4; j++ )
                    {
                        if ( contours_poly[i][j].x < contours_poly[i][0].x )
                            swap ( contours_poly[i][0], contours_poly[i][j] );
                        else
                            if ( contours_poly[i][j].y < contours_poly[i][1].y )
                                swap ( contours_poly[i][1], contours_poly[i][j] );
                            else
                                if ( contours_poly[i][j].x > contours_poly[i][2].x )
                                    swap ( contours_poly[i][2], contours_poly[i][j] );
                                else
                                    if ( contours_poly[i][j].y > contours_poly[i][3].y )
                                        swap ( contours_poly[i][3], contours_poly[i][j] );
                    }

                    boundRect[i] = boundingRect(contours_poly[i]);

                    rectangle( dst, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );

                    if (boundRect[i].width > boundRect[i].height) //dyanmic target
                    {
                        sprintf(str, "Width = %d", boundRect[i].width);
                        putText(dst, str,Point(5,90), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
                    }
                     else //static target
                    {
                        sprintf(str, "Height = %d", boundRect[i].height);
                        putText(dst, str,Point(5,110), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
                    }
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
                    cornerSubPix( img, localCorners, winSize, zeroZone, criteria );
                    corners.insert(corners.end(), localCorners.begin(), localCorners.end());

                    // test aspect ratio
                    Ratio_Top = length(pt[0].x, pt[0].y, pt[1].x, pt[1].y);
                    Ratio_Bottom = length(pt[2].x, pt[2].y, pt[3].x, pt[3].y);
                    Ratio_Left = length(pt[0].x, pt[0].y, pt[2].x, pt[2].y);
                    Ratio_Right = length(pt[1].x, pt[1].y, pt[3].x, pt[3].y);

                    if ( (Ratio_Left + Ratio_Right) < 10.0) //not a target
                    {
                        Ratio = 0;
                    }
                    else
                    {
                        Ratio = (Ratio_Top + Ratio_Bottom)/(Ratio_Left + Ratio_Right);
                    }

                    //what if we see all 4?
                    //                    if (contours.size() != 3)
                    //                    {
                    //                    if (contours_poly.size() < 4)
                    //                    {
                    //case 1 or 2
                    if (Ratio < 1) //subject to change
                    {
                        //contour is a tall and skinny one
                        //save off as static target
                        Static_Target.push_back(contours[i]);
                        drawContours(dst, contours_poly,i, Scalar(0,0,255), 3, 8, hierarchy, 0, Point() );
                        for( int i = 0; i < 4; i++ )
                        { cout<<" -- Static Target Original ["<<i<<"]  ("<<pt[i].x<<","<<pt[i].y<<")"<<endl; }

                    }
                    else
                    {
                        //contour is the short and wide, dynamic target
                        //save off as dynamic target
                        Dynamic_Target.push_back(contours[i]);
                        for( int i = 0; i < 4; i++ )
                        { cout<<" -- Dynamic Target Original ["<<i<<"]  ("<<pt[i].x<<","<<pt[i].y<<")"<<endl; }
                        drawContours(dst, contours_poly,i, Scalar(255, 0, 0), 3, 8, hierarchy, 0, Point() );

                    }
                    // }

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
        STOP_TIMING(FPS);
        time_now = 1000/GET_TIMING(FPS);
        time_ave = 1000/GET_AVERAGE_TIMING(FPS);
        sprintf(str, "Current FPS = %.f", time_now);
        putText(dst, str,Point(5,15), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
        sprintf(str, "Average FPS = %.f", time_ave);
        putText(dst, str,Point(5,30), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
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
        sprintf(str, "Targets S:%d D:%d", Static_Target.size(), Dynamic_Target.size());
        putText(dst, str,Point(5,60), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
        for (int i = 0; i < corners.size(); i++ ) {
            cout<<" -- Refined Corner ["<<i<<"]  ("<<corners[i].x<<","<<corners[i].y<<")"<<endl;
        }

        /// Show Images
        imshow("Raw", img);
        imshow("HSV", thresh);
        imshow("Final", dst);
        if (!USE_CAMERA) {
            waitKey(); // pause
            break;
        }
    } /// <---- End of While Loop (ESC has to be pressed to break out of loop) otherwise loop

    /// Destroy all windows and return 0 to end the program
    destroyAllWindows();
    return 0;
}
