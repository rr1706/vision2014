#include <iostream>

#include "detection.hpp"
#include "data.hpp"
#include "config.hpp"
#include "xyh.hpp"

using namespace std;
using namespace cv;

static char str[255];

void targetDetection(ThreadData &data)
{
    const float STATIC_TARGET_HEIGHT = 32.25;
    const float DYNAMIC_TARGET_HEIGHT = 4;
    const float COMBINED_TARGET_HEIGHT = 35.3;
    Mat img = data.image.clone();
    int IMAGE_WIDTH = img.cols, IMAGE_HEIGHT = img.rows;
    // Store the original image img to the Mat dst
    Mat dst = img.clone();

    // Convert image from input to threshold method
    cvtColor(img, img, CV_BGR2GRAY);
    if (displayMode == WindowMode::RAW) {
        Mat input = img.clone();
        cvtColor(input, input, CV_GRAY2RGB);
        Window::print("Ratchet Rockers 1706", input, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, input);
    }
    data.image = img;

    // "Threshold" image to pixels in the ranges
    threshold(img, img, gray_min, gray_max, CV_THRESH_BINARY);
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
    morphologyEx(img, img, MORPH_OPEN, kernel0, Point(-1, -1), dilations); // note replaced with open, idk if it will work here
    //erode(img, img, kernel1, Point(-1, -1), 1);
    //erode(img, img, kernel2, Point(-1, -1), 1);
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
    double imageHeightReal = 0.0;
    Mat contoursImg = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);
    double R[8] = {0};
    int P[CAMERA_COUNT][TARGET_COUNT];
    for (uint pi = 0; pi < CAMERA_COUNT; pi++) {
        for (uint pj = 0; pj < TARGET_COUNT; pj++) {
            P[pi][pj] = -1;
        }
    }
    vector<Target::Target> targets, staticTargets, dynamicTargets;
    TargetCase targetCase = NONE;

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
        Moments moment = moments(contour, false);
        Point2f massCenter(moment.m10/moment.m00, moment.m01/moment.m00);
        if (!isContourConvex(polygon)) {
            failedConvex++;
            RotatedRect minRect = minAreaRect(contour);
            Point2f rect_points[4];
            minRect.points(rect_points);
            double width = distance(rect_points[2], rect_points[3]);
            double height = distance(rect_points[2], rect_points[1]);
            double ratio = width / height;
            double areaRect = width * height;
            double areaContour = contourArea(contour);
            double deadSpace = areaContour / areaRect;
            if (ratio > 0.8 && ratio < 1.2 && ratio > 5 && ratio < .1) continue;
            if (deadSpace > 0.5) continue;
            Rect boundRect = boundingRect(contour);
            rectangle( dst, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0), 2, 8, 0 );

            int centerX = boundRect.x + boundRect.width / 2;
            int centerY = boundRect.y + boundRect.height / 2;
            Point2i center = {centerX, centerY};

            imageHeightReal = (IMAGE_HEIGHT * COMBINED_TARGET_HEIGHT) / boundRect.height;
            double Center_Static_X = (boundRect.x + (boundRect.width / 2)) - (IMAGE_WIDTH/2);
            double Plane_Distance_Combined = (imageHeightReal) / Tan_FOV_Y_Half;
            double In_Screen_Angle = (cameraInfo.fieldOfView.x / IMAGE_WIDTH) * Center_Static_X;
            double Real_Distance = Plane_Distance_Combined / (cos(In_Screen_Angle * CV_PI / 180));
            double Mod_Real_Distance = 0;
            if (Center_Static_X > 0)
                Mod_Real_Distance = Real_Distance + metersToInches(.5);
            if (ratio > 1) {
                targetCase = LEFT;
            } else {
                targetCase = RIGHT;
            }

            sprintf(str, "PLD:%.2fm RTO:%.2f CSX:%.2f", inchesToMeters(Plane_Distance_Combined), inchesToMeters(Real_Distance), inchesToMeters(Mod_Real_Distance));
            putText(dst, str, center + Point2i(-150, 15), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));

            for (int j = 0; j < 4; j++)
                line(dst, rect_points[j], rect_points[(j+1)%4], Scalar(0, 255, 255),2, 8);
            Moments moment = moments(contour, false);
            Point2i fakeDynamicCenter = boundRect.tl() + Point2i(boundRect.width / 2, 0);
            Point2i fakeStaticCenter = boundRect.tl() + Point2i(0, boundRect.height / 2);
            // lie about the values a little bit
            // TODO fake values for the corners
            Target::Target fakeDynamic = {Target::DYNAMIC, Real_Distance, Plane_Distance_Combined, moment, fakeDynamicCenter, fakeDynamicCenter, boundRect, minRect, In_Screen_Angle};
            Target::Target fakeStatic = {Target::STATIC, Real_Distance, Plane_Distance_Combined, moment, fakeStaticCenter, fakeStaticCenter, boundRect, minRect, In_Screen_Angle};
            targets.push_back(fakeDynamic);
            targets.push_back(fakeStatic);
            staticTargets.push_back(fakeStatic);
            dynamicTargets.push_back(fakeDynamic);
            continue;
        }
        if (false && polygon.size() != 4) {
            failedSides++;
            continue;
        }
        Rect boundRect = boundingRect(contour);
        RotatedRect minRect = minAreaRect( Mat(contour));
        Point2f rect_points[4];
        minRect.points(rect_points);
        for (int j = 0; j < 4; j++)
            line(dst, rect_points[j], rect_points[(j+1)%4], Scalar(0, 255, 0),2, 8);
        rectangle(dst, boundRect, Scalar(0, 255, 0));
        int centerX = boundRect.x + boundRect.width / 2;
        int centerY = boundRect.y + boundRect.height / 2;
        Point2i center = {centerX, centerY};
        if (center.x > 450)
        {
            continue;
        }
        if (center.y < 50)
        {
            continue;
        }

        // ratio helps determine orientation of rectangle (vertical / horizontal)
        double ratio = static_cast<double>(boundRect.width) / static_cast<double>(boundRect.height);
        if (isAlmostSquare(ratio)) {
            failedSquare++;
            sprintf(str, "Failed Square R:%.2f", ratio);
            putText(dst, str, center, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(0, 0, 255));
            continue; // go to next contour
        } else if (isExtraLong(ratio)) {
            failedVLarge++;
            continue;
        }

        // test aspect ratio
        success++;
        Target::Type targetType;
        if (boundRect.height > (boundRect.width * 2)) {
            targetType = Target::STATIC;
        } else if (boundRect.width > (boundRect.height * 2)) {
            targetType = Target::DYNAMIC;
        } else {
            sprintf(str, "Fail T:Type %.2f", ratio);
            putText(dst, str, center, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(0, 0, 255));
            continue;
        }
        double planeDistance = 0, realDistance = 0, inScreenAngle = 0;

        if (targetType == Target::STATIC) // static target
        {
            RotatedRect minRect = minAreaRect(contour);

            imageHeightReal = (IMAGE_HEIGHT * STATIC_TARGET_HEIGHT) / minRect.boundingRect().height;
            //  refinedHeight = distance(localCorners[0], localCorners[2]);
            //  flatHeight = localCorners[2].y - localCorners[0].y;        int targetId = 0; // TODO find this, 0-7

            double Center_Static_X = (boundRect.x + (boundRect.width / 2)) - (IMAGE_WIDTH/2);
            double Plane_Distance = (imageHeightReal) / Tan_FOV_Y_Half;
            inScreenAngle = (cameraInfo.fieldOfView.x / IMAGE_WIDTH) * Center_Static_X;
            double Real_Distance = Plane_Distance / (cos(inScreenAngle * CV_PI / 180));
            double modifiedReal = Real_Distance;
            if (Real_Distance > 149) {
                modifiedReal = Real_Distance + metersToInches(0.72583);
            } else {
                modifiedReal = Real_Distance + metersToInches(0.216);
            }
            planeDistance = Plane_Distance;
            realDistance = modifiedReal;

            if (Plane_Distance < 5) {
                continue;
            }

            sprintf(str, "H:%d W:%d", boundRect.height, boundRect.width);
            putText(dst, str, center, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(100, 100, 255));
            sprintf(str, "P:%.2fm R:%.2f M:%.2f", inchesToMeters(Plane_Distance), inchesToMeters(Real_Distance), inchesToMeters(modifiedReal));
            putText(dst, str, center + Point2i(0, 15), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            //contour is a tall and skinny one
            //save off as static target
            Static_Target.push_back(contours[i]);
        }
        else if (targetType == Target::DYNAMIC)
        {
            if (imageHeightReal == 0.0) // only set with dynamic if there is no value, static is probably more accurate
                imageHeightReal = (IMAGE_HEIGHT * DYNAMIC_TARGET_HEIGHT) / boundRect.height;
            double Center_Static_X = (boundRect.x + (boundRect.width / 2)) - (IMAGE_WIDTH/2);
            double Plane_Distance_Dynamic = (imageHeightReal) / Tan_FOV_Y_Half;
            inScreenAngle = (cameraInfo.fieldOfView.x / IMAGE_WIDTH) * Center_Static_X;
            double Real_Distance_Dynamic = Plane_Distance_Dynamic / (cos(inScreenAngle * CV_PI / 180));
            planeDistance = Plane_Distance_Dynamic;
            realDistance = Real_Distance_Dynamic;

            if (Plane_Distance_Dynamic < 5) {
                continue;
            }

            sprintf(str, "BRH:%d", boundRect.height);
            putText(dst, str, center, CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));
            sprintf(str, "P:%.2fm R:%.2f M:%.2f", inchesToMeters(Plane_Distance_Dynamic), inchesToMeters(Real_Distance_Dynamic), 0.0);
            putText(dst, str, center + Point2i(0, 15), CV_FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 100, 100));


            //contour is the short and wide, dynamic target
            //save off as dynamic target
            Dynamic_Target.push_back(contours[i]);
        }

        Target::Target target = {targetType, realDistance, planeDistance, moment, massCenter, center, boundRect, minRect, inScreenAngle};
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
#ifdef PRINT_TEST
    cout << "Total: " << totalContours << " | Failures Area: " << failedArea << " Hierarchy: " << failedHierarchy <<
            " Sides: " << failedSides << " Convex: " << failedConvex << " Square: " << failedSquare << " VeryLarge: " << failedVLarge << " | Success: " << success << endl;
#endif
    sortTargets(targets);
    sortTargets(dynamicTargets);
    sortTargets(staticTargets);
    data.targets = targets;
    data.dynamicTargets = dynamicTargets;
    data.staticTargets = staticTargets;
    if (targetCase == NONE && dynamicTargets.size() > 0 && staticTargets.size() > 0 && targets.size() == 2
            && staticTargets[0].massCenter.x > dynamicTargets[0].massCenter.x) {
        //case left
        targetCase = LEFT;
        R[0] = dynamicTargets[0].realDistance;
        R[4] = staticTargets[0].realDistance;
        P[0][4] = staticTargets[0].massCenter.x;
        P[0][0] = dynamicTargets[0].massCenter.x;
    } else if (targetCase == NONE && dynamicTargets.size() > 0 && staticTargets.size() > 0 && targets.size() == 2) {
        //case right
        targetCase = RIGHT;
        R[4] = dynamicTargets[0].realDistance;
        R[0] = staticTargets[0].realDistance;
        P[0][4] = staticTargets[0].massCenter.x;
        P[0][0] = dynamicTargets[0].massCenter.x;
    } else if (staticTargets.size() >= 2 && dynamicTargets.size() >= 2) {
        if (staticTargets[0].massCenter.x < dynamicTargets[0].massCenter.x)
        {
            targetCase = ALL;
            R[4] = staticTargets[0].realDistance;
            R[0] = dynamicTargets[0].realDistance;
            R[5] = staticTargets[1].realDistance;
            R[1] = dynamicTargets[1].realDistance;
            P[0][4] = staticTargets[0].massCenter.x;
            P[0][0] = dynamicTargets[0].massCenter.x;
            P[0][5] = staticTargets[1].massCenter.x;
            P[0][1] = dynamicTargets[1].massCenter.x;
        }
        else
        {
            targetCase = ALL_INVERTED;
            R[1] = staticTargets[1].realDistance;
            R[4] = dynamicTargets[1].realDistance;
            R[5] = staticTargets[0].realDistance;
            R[1] = dynamicTargets[0].realDistance;
            P[0][0] = staticTargets[1].massCenter.x;
            P[0][4] = dynamicTargets[1].massCenter.x;
            P[0][5] = staticTargets[0].massCenter.x;
            P[0][1] = dynamicTargets[0].massCenter.x;
        }
    } else if (staticTargets.size() == 2 && dynamicTargets.size() == 0) {
        targetCase = ALL;
    } else if (staticTargets.size() == 0 && dynamicTargets.size() == 2) {
        targetCase = ALL;
    } else if (staticTargets.size() == 2 && dynamicTargets.size() == 1) {
        targetCase = ALL;
    } else if (staticTargets.size() == 1 && dynamicTargets.size() == 2) {
        targetCase = ALL;
    }
    data.pairCase = targetCase;
    double xPos, yPos, heading;
    FindXYH(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], P, xPos, yPos, heading);
    // NOTE: THIS IS PER CAMERA FOR DEMO ONLY. Please look to the sa() function above for the FindXYH that is actually used.
    if (staticTargets.size() > 0) {
        Target::Target target = staticTargets[0];
        data.targetLog.log("distance", target.planeDistance);
        data.targetLog.log("bound_height", target.boundRect.height);
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
    sprintf(str, "Targets S:%ld D:%ld", (long) staticTargets.size(), (long) dynamicTargets.size());
    putText(dst, str,Point(5,60), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    sprintf(str, "Image Height %dpx %.2fin", IMAGE_HEIGHT, imageHeightReal);
    putText(dst, str,Point(5,75), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    sprintf(str, "Heading %f xR %f yR %f", heading, xPos, yPos);
    putText(dst, str,Point(5,90), CV_FONT_HERSHEY_COMPLEX_SMALL, 0.75, Scalar(255,0,255),1,8,false);
    applyText(statusText, Point(5, 90), dst);
    //draw crosshairs
    //line(dst, Point( IMAGE_WIDTH/2, 0), Point(IMAGE_WIDTH / 2, IMAGE_HEIGHT), Scalar(0, 255, 255), 1, 8, 0);
    //line(dst, Point( 0, IMAGE_HEIGHT/2), Point(IMAGE_WIDTH, IMAGE_HEIGHT/2), Scalar(0, 255, 255), 1, 8, 0);
    /// Show Images
    if (displayMode == WindowMode::FINAL && procMode == DEMO) {
        WindowMode::print(displayMode, dst);
        Window::print("Ratchet Rockers 1706", dst, Point(IMAGE_WIDTH - 200, 15));
        imshow(windowName, dst);
    }
    data.dst = dst;
}
