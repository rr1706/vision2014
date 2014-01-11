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
#define LOAD_IMAGE 1
#define USE_CAMERA 0

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
    /// Declare default values of trackbars
    int hue_min = 70;
    int saturation_min = 30;
    int value_min = 155;
    int hue_max = 255;
    int saturation_max = 255;
    int value_max = 255;
    const int accuracy = 2; //maximum distance between the original curve and its approximation

    /// Declare Timing and create variables to store frame times
    DECLARE_TIMING(FPS);
    double time_now;
    double time_ave;

    /// Define character to store a string of up to 50 characters, to be printed on the image
    char str[50];

    /// Initialize Camera and cv::Mat to hold images
    CvCapture* capture0 = cvCreateCameraCapture(0);
    Mat img, dst, thresh;
    assert( capture0);

    /// Initialize a font, used for printed text onto an image
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, 0.75, 0, 1, CV_AA);

    /// Create Windows
    namedWindow("Raw", CV_WINDOW_AUTOSIZE);
    namedWindow("HSV", CV_WINDOW_AUTOSIZE);
    namedWindow("Final", CV_WINDOW_AUTOSIZE);

    Mat inframe;

    if(LOAD_IMAGE)
    {
        inframe = imread("raw_img.jpg");
    }

#if ( USE_CAMERA == 1 )
    while ( 1 )
    {
        /// Start timing a frame (FPS will be a measurement of the time it takes to process all the code for each frame)
        START_TIMING(FPS);

        char ch;
        //Break out of loop if esc is pressed
        if( (ch = waitKey(10) ) == ESC)
            break;

        /// Grab a frame and contain it in the cv::Mat img
        img = cvQueryFrame( capture0 );
        if ( ch == 's' )
            imwrite ( "raw_img.jpg", img );
#else
        //changes according to camera used
        inframe.copyTo(img);
        {


        /// Store the original image img to the Mat dst
        img.copyTo(dst);
#endif
        /// Convert img from BGR(Blue,Green,Red) to HSV(Hue,Saturation,Value)
        cvtColor(img,img,CV_BGR2HSV);

        /// "Threshold" image to pixels in the ranges
        inRange(img,Scalar(hue_min, saturation_min, value_min), Scalar(hue_max, saturation_max, value_max),img);

        /// Save img to thresh Mat to show to the user
        img.copyTo(thresh);

        erode(img, img, NULL);

        /// Declare containers for contours and contour heirarchy
        vector<vector<Point> > contours;
        vector<vector<Point> > Static_Target;
        vector<vector<Point> > Dynamic_Target;
        vector<Vec4i> hierarchy;

        findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );

        /// Declare container for approximated polygons
        vector<vector<Point> > contours_poly( contours.size() );

        double Ratio;
        double Ratio_Top;
        double Ratio_Bottom;
        double Ratio_Right;
        double Ratio_Left;

        /// Create a for loop to go through each contour (i) one at a time
        //why not while(contours)?
        for( unsigned int i = 0; i < contours.size(); i++ )
        {
            /// If contour has an area greater than 500 pixels
            if(contourArea(contours[i])>Minimum_Area)
            {
                /// Approximate a Polygon and save to contours_poly
                approxPolyDP( contours[i], contours_poly[i], accuracy, true );

                /// If 4 sided
                if(contours_poly[i].size()== 4 && isContourConvex(contours_poly[i]))
                {

                    //drawContours(dst, contours_poly,i, Scalar(255,0,0), 3, 8, hierarchy, 0, Point() );

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


                    /// test aspect ratio
                    Ratio_Top = length(contours_poly[i][0].x, contours_poly[i][0].y, contours_poly[i][1].x, contours_poly[i][1].y);
                    Ratio_Bottom = length(contours_poly[i][2].x, contours_poly[i][2].y, contours_poly[i][3].x, contours_poly[i][3].y);
                    Ratio_Left = length(contours_poly[i][0].x, contours_poly[i][0].y, contours_poly[i][2].x, contours_poly[i][2].y);
                    Ratio_Right = length(contours_poly[i][1].x, contours_poly[i][1].y, contours_poly[i][3].x, contours_poly[i][3].y);

                    if ( (Ratio_Left + Ratio_Right) < 10.0) //not a target
                    {
                        Ratio = 0;
                    }
                    else
                    {
                        Ratio = (Ratio_Left + Ratio_Right)/(Ratio_Top + Ratio_Bottom);
                    }

                    sprintf(str, "Ratio = %lf", Ratio);
                    putText(dst, str,Point(5,50), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);

                    //what if we see all 4?
//                    if (contours.size() != 3)
//                    {
//                        if (contours.size() < 4)
//                        {
                            //case 1 or 2
                            if (Ratio > 1) //subject to change
                            {
                                //contour is a tall, skinny one
                                //save off as stationary target
                                Static_Target.push_back(contours[i]);
                                drawContours(dst, contours_poly,i, Scalar(0,0,255), 3, 8, hierarchy, 0, Point() );

                            }
                            else
                            {
                                //contour is the short, long, dynamic target
                                //save off as dynamic target
                                Dynamic_Target.push_back(contours[i]);
                                drawContours(dst, contours_poly,i, Scalar(255, 0, 0), 3, 8, hierarchy, 0, Point() );

                            }


                    /// Order targets left to right

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
        if(Mass_Center_Static.size() > 0 && Mass_Center_Dynamic.size() > 0
                && Mass_Center_Static[0].x > Mass_Center_Dynamic[0].x)
        {
            //case left
        }

        else
        {
            //case right
        }

        /// Stop timing and calculate FPS and Average FPS
        STOP_TIMING(FPS);
        time_now = 1000/GET_TIMING(FPS);
        time_ave = 1000/GET_AVERAGE_TIMING(FPS);
        sprintf(str, "Current FPS = %.f", time_now);
        putText(dst, str,Point(5,15), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
        sprintf(str, "Average FPS = %.f", time_ave);
        putText(dst, str,Point(5,30), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);


        /// Show Images
        imshow("Raw", img);
        imshow("HSV", thresh);
        imshow("Final", dst);

#if ( USE_CAMERA == 0 )
        waitKey();
#endif

    } /// <---- End of While Loop (ESC has to be pressed to break out of loop) otherwise loop

    /// Destroy all windows and return 0 to end the program
    destroyAllWindows();
    return 0;
}
