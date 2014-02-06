/*
Version 16 - Added check for not to try and compute heading unless you have both the P and R for a target.

Find the Robot Coordinates (xR, yR, and Heading) with respect to the corner of the playing field as the origin.

You will call a function:

    FindXYH(R1, R2, R3, R4, R5, R6, R7, R8, P[i][t])

The R values are the range in inches to the center of the targets.
P[i][t] is a 3 by 8 matrix to give the column pixel of the center of the target for each camera (i=1,2,3) and each target (t=1,2,3,4,5,6,7,8).
(Note: Take care.  A 3x8 array is really from (0,1,2) by (0,1,2,3,4,5,6,7) so just make your array P[4][9].  Memory is cheap, errors are not).

If you don’t have a value for a Range, enter it as zero.  If you don’t have a pixel for a camera and target pair, enter it as -1.

This function will then generate (xR, yR, H) in inches and degrees.  If multiple solutions are available, they will be averaged.
*/

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include "iomanip"
#include "xyh.hpp"
using namespace std;


//Function Prototypes

//Declaring Global Variables
//Used for Robot Locations
static double xR;
static double yR;
static double H;

/*
 *These are the cooindHeadirindates for the center of the hortizontal targets with respect to the field origin.
 *Making them global as they are used in both FindXY and FindHeading.
*/
static double x1t=-6.74;
static double y1t=20.88;
static double x2t=-6.74;
static double y2t=275.12;
static double x3t=654.74;
static double y3t=275.12;
static double x4t=654.74;
static double y4t=20.88;
static double x5t=-0.28;
static double y5t=39.63;
static double x6t=-0.28;
static double y6t=256.37;
static double x7t=648.28;
static double y7t=256.37;
static double x8t=648.28;
static double y8t=39.63;

/*
 *These hold the plus and minus solutions until you can do checks to determine which solution is valid.
 *Making them global as they are used in both FindXY and FindHeading.
*/
static double xPlus;
static double xMinus;
static double yPlus;
static double yMinus;

std::ofstream ofs ("xyh.log", std::ofstream::out);

int test()
{

    double TestR1, TestR2, TestR3, TestR4, TestR5, TestR6, TestR7, TestR8;
    int P[3][8];
    int i,j;


    // Example:
    TestR1=180;
    TestR2=240;
    TestR3=TestR4=TestR5=TestR6=TestR7=TestR8=0;
    for (i=0; i<3; i++)
       for (j=0; j<8; j++)
           P[i][j]=-1;
    P[0][1]=170;



    double xpos, ypos, heading;
    // This is what Hunter and Connor will call.  It will return xR and yR (inches) and Heading (degrees) as global variables.
    FindXYH(TestR1, TestR2, TestR3, TestR4, TestR5, TestR6, TestR7, TestR8, P, xpos, ypos, heading);

    return(0);
}



/*
 * This function finds global variables xR and yR and Heading of the robot.
 * Enter the ranges to the 4 hortizontal targets (R1 to R4) and to the 4 vertical targets (R5 to R8) in inches and
 * an array P[i][t] which is a 3 by 8 matrix to give the column pixel of the center of the targets for each
 * camera (i=1,2,3) and each target (t=1,2,3,4,5,6,7,9).
 * To keep it simple, just make the array P[4][9] so can keep tract of subscripts.
 * If you don’t have a value for a Range, enter it as zero.
 * If you don’t have a pixel for a camera and target pair, enter it as -1.
*/
void FindXYH(double R1, double R2, double R3, double R4, double R5, double R6, double R7, double R8, int P[3][8], double &xPos, double &yPos, double &heading)
{
    FindXY(R1, R2, R3, R4, R5, R6, R7, R8);
    FindHeading(R1, R2, R3, R4, R5, R6, R7, R8, P);
    xPos = xR;
    yPos = yR;
    heading = H;
}

/*
 * This function finds global variables xR and yR of the robot.
 * Enter the ranges to the 4 hortizontal targets (R1 to R4) and to the 4 vertical targets (R5 to R8) in inches.
 * Everything is with respect to the origin at the corner of the field.
*/
void FindXY(double R1, double R2, double R3, double R4, double R5, double R6, double R7, double R8)
{

    // These are used to average if can get more than one independent solution.
    double ix=0;
    double iy=0;
    double xsum=0;
    double ysum=0;

    // Have no R values so just return.
    if (R1<=0 && R2<=0 && R3<=0 && R4<=0)
        return;

    ofs<<"R Values    (xPlus,yPlus)   (xMinus,yMinus)  Selected"<<endl;
    ofs<<fixed<<showpoint;
    ofs<<setprecision(2);

// FIRST LOOK FOR SOLUTIONS USING ADJACENT CORNERS.

    // HORIZONTAL TARGETS: There are 4 opportunities to find a solutions using targets at adjacent corners.
    if (R1>0 && R2>0)
    {
        ofs<<"R1 and R2";
        FindPlusMinusSolutionInFieldCooridinates(R1, R2, x1t, y1t, x2t, y2t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R1>0 && R4>0)
    {
        ofs<<"R1 and R4";
        FindPlusMinusSolutionInFieldCooridinates(R1, R4, x1t, y1t, x4t, y4t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R2>0 && R3>0)
    {
        ofs<<"R2 and R3";
        FindPlusMinusSolutionInFieldCooridinates(R2, R3, x2t, y2t, x3t, y3t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R3>0 && R4>0)
    {
        ofs<<"R3 and R4";
        FindPlusMinusSolutionInFieldCooridinates(R3, R4, x3t, y3t, x4t, y4t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }

    // VERTICAL TARGETS: There are 4 opportunities to find a solutions using targets at adjacent corners.
    if (R5>0 && R6>0)
    {
        ofs<<"R5 and R6";
        FindPlusMinusSolutionInFieldCooridinates(R5, R6, x5t, y5t, x6t, y6t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R5>0 && R8>0)
    {
        ofs<<"R5 and R8";
        FindPlusMinusSolutionInFieldCooridinates(R5, R8, x5t, y5t, x8t, y8t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R6>0 && R7>0)
    {
        ofs<<"R6 and R7";
        FindPlusMinusSolutionInFieldCooridinates(R6, R7, x6t, y6t, x7t, y7t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R7>0 && R8>0)
    {
        ofs<<"R7 and R8";
        FindPlusMinusSolutionInFieldCooridinates(R7, R8, x7t, y7t, x8t, y8t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }

 // MIX HORIZONTAL AND VERTICAL TARGETS: There are 8 opportunities to find a solutions using targets at adjacent corners.
    if (R1>0 && R6>0)
    {
        ofs<<"R1 and R6";
        FindPlusMinusSolutionInFieldCooridinates(R1, R6, x1t, y1t, x6t, y6t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R1>0 && R8>0)
    {
        ofs<<"R1 and R8";
        FindPlusMinusSolutionInFieldCooridinates(R1, R8, x1t, y1t, x8t, y8t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R2>0 && R5>0)
    {
        ofs<<"R2 and R5";
        FindPlusMinusSolutionInFieldCooridinates(R2, R5, x2t, y2t, x5t, y5t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R2>0 && R7>0)
    {
        ofs<<"R2 and R7";
        FindPlusMinusSolutionInFieldCooridinates(R2, R7, x2t, y2t, x7t, y7t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R3>0 && R6>0)
    {
        ofs<<"R3 and R6";
        FindPlusMinusSolutionInFieldCooridinates(R3, R6, x3t, y3t, x6t, y6t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R3>0 && R8>0)
    {
        ofs<<"R3 and R8";
        FindPlusMinusSolutionInFieldCooridinates(R3, R8, x3t, y3t, x8t, y8t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R4>0 && R5>0)
    {
        ofs<<"R4 and R5";
        FindPlusMinusSolutionInFieldCooridinates(R4, R5, x4t, y4t, x5t, y5t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }
    if (R4>0 && R7>0)
    {
        ofs<<"R4 and R7";
        FindPlusMinusSolutionInFieldCooridinates(R4, R7, x4t, y4t, x7t, y7t);
        ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
        SelectSolutionForAdjacentCorners();
        ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
        ix++;
        xsum=xsum+xR;
        iy++;
        ysum=ysum+yR;
    }


// NOW LOOK FOR SOLUTIONS USING OPPOSITE CORNERS.
// FOR OPPOSITE CORNERS, YOU GET TWO SOLUTIONS IN THE PLAYING FIELD.
// HENCE, YOU NEED MORE INFORMATION TO FIGURE OUT WHICH SOLUTION TO USE.
// ONE WAY TO DO THIS IS TO USE GYRO AND ENCODER DATA (THIS ISN'T READY YET).
// THE OTHER WAY TO DO THIS IS, IF FOUND A SOLUTION USING ADJACENT CORNERS, USE THAT
// INFORMATION TO PICK THE CORRECT SOLUTION WHEN USING OPPOSITE CORNERS.  TO DO THIS,
// YOU MUST HAVE AT LEAST ONE SOLUTION USING ADJACENT CORNERS (ix>0).

     // VERTICAL TARGETS: There are 2 opportunities to find a solutions using targets at opposite corners.
    if (ix>0)
    {
        if (R1>0 && R3>0)
        {
            ofs<<"R1 and R3";
            FindPlusMinusSolutionInFieldCooridinates(R1, R3, x1t, y1t, x3t, y3t);
            ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
            SelectSolutionForOppositeCorners(ix, iy, xsum, ysum);
            ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
            ix++;
            xsum=xsum+xR;
            iy++;
            ysum=ysum+yR;
        }
        if (R2>0 && R4>0)
        {
            ofs<<"R2 and R4";
            FindPlusMinusSolutionInFieldCooridinates(R2, R4, x2t, y2t, x4t, y4t);
            ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
            SelectSolutionForOppositeCorners(ix, iy, xsum, ysum);
            ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
            ix++;
            xsum=xsum+xR;
            iy++;
            ysum=ysum+yR;
        }

        // HORIZONTAL TARGETS: There are 2 opportunities to find a solutions using targets at opposite corners.
        if (R5>0 && R7>0)
        {
            ofs<<"R5 and R7";
            FindPlusMinusSolutionInFieldCooridinates(R5, R7, x5t, y5t, x7t, y7t);
            ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
            SelectSolutionForOppositeCorners(ix, iy, xsum, ysum);
            ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
            ix++;
            xsum=xsum+xR;
            iy++;
            ysum=ysum+yR;
        }
        if (R6>0 && R8>0)
        {
            ofs<<"R6 and R8";
            FindPlusMinusSolutionInFieldCooridinates(R6, R8, x6t, y6t, x8t, y8t);
            ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
            SelectSolutionForOppositeCorners(ix, iy, xsum, ysum);
            ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
            ix++;
            xsum=xsum+xR;
            iy++;
            ysum=ysum+yR;
        }

        // MIX OF HORIZONTAL AND VERTICAL TARGETS: There are 4 opportunities to find a solutions using targets at opposite corners.
        if (R1>0 && R7>0)
        {
            ofs<<"R1 and R7";
            FindPlusMinusSolutionInFieldCooridinates(R1, R7, x1t, y1t, x7t, y7t);
            ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
            SelectSolutionForOppositeCorners(ix, iy, xsum, ysum);
            ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
            ix++;
            xsum=xsum+xR;
            iy++;
            ysum=ysum+yR;
        }
        if (R2>0 && R8>0)
        {
            ofs<<"R2 and R8";
            FindPlusMinusSolutionInFieldCooridinates(R2, R8, x2t, y2t, x8t, y8t);
            ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
            SelectSolutionForOppositeCorners(ix, iy, xsum, ysum);
            ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
            ix++;
            xsum=xsum+xR;
            iy++;
            ysum=ysum+yR;
        }
        if (R3>0 && R5>0)
        {
            ofs<<"R3 and R5";
            FindPlusMinusSolutionInFieldCooridinates(R3, R5, x3t, y3t, x5t, y5t);
            ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
            SelectSolutionForOppositeCorners(ix, iy, xsum, ysum);
            ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
            ix++;
            xsum=xsum+xR;
            iy++;
            ysum=ysum+yR;
        }
        if (R4>0 && R6>0)
        {
            ofs<<"R4 and R6";
            FindPlusMinusSolutionInFieldCooridinates(R4, R6, x4t, y4t, x6t, y6t);
            ofs<<"    ("<<xPlus<<","<<yPlus<<")"<<"("<<xMinus<<","<<yMinus<<")";
            SelectSolutionForOppositeCorners(ix, iy, xsum, ysum);
            ofs<<"    ("<<xR<<","<<yR<<")"<<endl;
            ix++;
            xsum=xsum+xR;
            iy++;
            ysum=ysum+yR;
        }
    }

    // If found more than one independent solution, average them.
    if (ix>0)
    {
        xR=xsum/ix;
        yR=ysum/iy;
    }
    ofs<<"The average of all solutions is "<<"("<<xR<<","<<yR<<")"<<endl;

}

/*
 *This function finds the plus and minus solution for one pair of ranges.
 *It returns the global variables xPlus, yPlus, xMinus, yMinus.
 *Before it returns these values, it converts them from the origin being at the center of the horizontal target
 *to the origin being at the corner of the playing field.
*/
void FindPlusMinusSolutionInFieldCooridinates(double A, double B, double a, double b, double c, double d)
{
    double E, F, D, G, H, I;
    A=pow(A,2);
    B=pow(B,2);

//For when c=a (Pencil Paper)
if (c==a)
    {
        G=(2*a-2*c)/(2*d-2*b);
        H=(A-B-pow(a,2)-pow(b,2)+pow(c,2)+pow(d,2))/(2*d-2*b);
        I=H-b;
        xPlus=((-(2*G*I-2*a)+sqrt(pow((2*G*I-2*a),2)-4*(pow(G,2)+1)*(pow(I,2)+pow(a,2)-A)))/(2*(pow(G,2)+1)));
        xMinus=((-(2*G*I-2*a)-sqrt(pow((2*G*I-2*a),2)-4*(pow(G,2)+1)*(pow(I,2)+pow(a,2)-A)))/(2*(pow(G,2)+1)));
        yPlus=G*xPlus+H;
        yMinus=G*xMinus+H;
    }
    
//For when d=b (Pen Paper)
    if (d==b)
    {
        D=(2*b-2*d)/(2*c-2*a);
        E=(A-B-pow(a,2)-pow(b,2)+pow(c,2)+pow(d,2))/(2*c-2*a);
        F=E-a;
        yPlus=((-(2*D*F-2*b)+sqrt(pow((2*D*F-2*b),2)-4*(pow(D,2)+1)*(pow(F,2)+pow(b,2)-A)))/(2*(pow(D,2)+1)));
        yMinus=((-(2*D*F-2*b)-sqrt(pow((2*D*F-2*b),2)-4*(pow(D,2)+1)*(pow(F,2)+pow(b,2)-A)))/(2*(pow(D,2)+1)));
        xPlus=D*yPlus+E;
        xMinus=D*yMinus+E;
    }

//For when c!=a and d!=b (you can use either form as no divide by zero)
    if (c!=a && d!=b)
    {
        D=(2*b-2*d)/(2*c-2*a);
        E=(A-B-pow(a,2)-pow(b,2)+pow(c,2)+pow(d,2))/(2*c-2*a);
        F=E-a;
        yPlus=((-(2*D*F-2*b)+sqrt(pow((2*D*F-2*b),2)-4*(pow(D,2)+1)*(pow(F,2)+pow(b,2)-A)))/(2*(pow(D,2)+1)));
        yMinus=((-(2*D*F-2*b)-sqrt(pow((2*D*F-2*b),2)-4*(pow(D,2)+1)*(pow(F,2)+pow(b,2)-A)))/(2*(pow(D,2)+1)));
        xPlus=D*yPlus+E;
        xMinus=D*yMinus+E;
    }

}

/*
 *This function is for solutions when using two ranges from targets at adjacent corners.
 *It simply uses the boundaries of the field to determine which solution (plus or minus) is the real solution.
*/
void SelectSolutionForAdjacentCorners()
{
    //Stipulations for results to keep them within bounds of playing field
    if(xPlus>=0 && yPlus>=0 && xPlus<=648 && yPlus<=296)
    {
        xR=xPlus;
        yR=yPlus;
    }
    if(xMinus>=0 && yMinus>=0 && xMinus<=648 && yMinus<=296)
    {
        xR=xMinus;
        yR=yMinus;
    }
}

/*
 *This function is for solutions when using two ranges from targets at opposite corners.
 *It requires additonal data to determine which solution is correct.  There are two ways to do this.
 *  1. If you already have a solution from targets at adjacent corners, use it to find the correct solution.
 *  2. If have update location information from the gyro and encoders, you can use the information (this isn't done yet).
*/
void SelectSolutionForOppositeCorners(double ixl, double iyl, double xsuml, double ysuml)
{
    double xRl, yRl;
    double DistancePlus, DistanceMinus;

    // If already have more than one independent solution, average them.
    if (ixl>0)
    {
        xRl=xsuml/ixl;
        yRl=ysuml/iyl;
    }
    else
    {
        cerr << "Wrong case in SelectSOlutionForOppositeCorners (SHOULD NOT HAPPEN, THE WORLD HATES YOU)" << endl;
        exit(1);
    }

    /*
     *Now use xRl and yRl to pick which solution (plus or minus) is real.
     *To do this, find which solution (plus or minus) is closer to xRl and yRl
    */
    DistancePlus=sqrt(pow((xRl-xPlus),2)+pow((yRl-yPlus),2));
    DistanceMinus=sqrt(pow((xRl-xMinus),2)+pow((yRl-yMinus),2));
    if (DistancePlus<DistanceMinus)
    {
        xR=xPlus;
        yR=yPlus;
    }
    else
    {
        xR=xMinus;
        yR=yMinus;
    }

    // Do not have gyro or encoder info yet so just use method 1 described above.
}

/*
 *This function finds the heading (H) when passed the robots (xR,yR) in inches and an array P[i][t].
 *P[i,t] is a 3 by 8 matrix to give the column pixel of the center of the targets for each
 *camera (i=1,2,3) and each target (t=1,2,3,4,5,6,7,9).
 *To keep it simple, just make the array P[4][9] so can keep tract of subscripts.
 *If you don’t have a value for a Range, enter it as zero.
 *If you don’t have a pixel for a camera and target pair, enter it as -1.
 *
 *Note:  No correction to heading are needed when you change origins from horizontal target to the corner of the field.
*/

void FindHeading(double R1, double R2, double R3, double R4, double R5, double R6, double R7, double R8, int arr[3][8])
{
    int P[4][9];
    for (uint i = 0; i < 3; i++) {
        for (uint j = 0; j < 8; j++) {
           P[i + 1][j + 1] = arr[i][j];
        }
    }
    //Use to average if have more than one solution for heading.
    double Hsum=0;
    int iH=0;

    // Show what was sent.
    int i,j;
    for (i=1; i<=3; i++)
        for (j=1; j<=8; j++)
            ofs<<"P["<<i<<"]["<<j<<"]="<<P[i][j]<<endl;


// CHECK CAMERA 1 SEES TARGETS 1 to 8
    if (P[1][1]>=0 && R1>0) //If Camera 1 sees Target 1:
    {
        iH++;
        H = -60 - (180/3.14159)*asin((yR-y1t)/R1) - (P[1][1] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[1][2]>=0 && R2>0) //If Camera 1 sees Target 2:
    {
        iH++;
        H = -60 + (180/3.14159)*asin((y2t-yR)/R2) - (P[1][2] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[1][3]>=0 && R3>0) //If Camera 1 sees Target 3:
    {
        iH++;
        H = +120 - (180/3.14159)*asin((y3t-yR)/R3) - (P[1][3] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[1][4]>=0 && R4>0) //If Camera 1 sees Target 4:
    {
        iH++;
        H = +120 + (180/3.14159)*asin((yR-y4t)/R4) - (P[1][4] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[1][5]>=0 && R5>0) //If Camera 1 sees Target 5:
    {
        iH++;
        H = -60 - (180/3.14159)*asin((yR-y5t)/R5) - (P[1][5] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[1][6]>=0 && R6>0) //If Camera 1 sees Target 6:
    {
        iH++;
        H = -60 + (180/3.14159)*asin((y6t-yR)/R6) - (P[1][6] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[1][7]>=0 && R7>0) //If Camera 1 sees Target 7:
    {
        iH++;
        H = +120 - (180/3.14159)*asin((y7t-yR)/R7) - (P[1][7] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[1][8]>=0 && R8>0) //If Camera 1 sees Target 8:
    {
        iH++;
        H = +120 + (180/3.14159)*asin((yR-y8t)/R8) - (P[1][8] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }


// CHECK CAMERA 2 SEES TARGETS 1 to 8
   if (P[2][1]>=0 && R1>0) //If Camera 2 sees Target 1:
    {
        iH++;
        H = -180 - (180/3.14159)*asin((yR-y1t)/R1) - (P[2][1] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[2][2]>=0 && R2>0) //If Camera 2 sees Target 2:
    {
        iH++;
        H = -180 + (180/3.14159)*asin((y2t-yR)/R2) - (P[2][2] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[2][3]>=0 && R3>0) //If Camera 2 sees Target 3:
    {
        iH++;
        H = 0 - (180/3.14159)*asin((y3t-yR)/R3) - (P[2][3] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[2][4]>=0 && R4>0) //If Camera 2 sees Target 4:
    {
        iH++;
        H = 0 + (180/3.14159)*asin((yR-y4t)/R4) - (P[2][4] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[2][5]>=0 && R5>0) //If Camera 2 sees Target 5:
    {
        iH++;
        H = -180 - (180/3.14159)*asin((yR-y5t)/R5) - (P[2][5] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[2][6]>=0 && R6>0) //If Camera 2 sees Target 6:
    {
        iH++;
        H = -180 + (180/3.14159)*asin((y6t-yR)/R6) - (P[2][6] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[2][7]>=0 && R7>0) //If Camera 2 sees Target 7:
    {
        iH++;
        H = 0 - (180/3.14159)*asin((y7t-yR)/R7) - (P[2][7] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[2][8]>=0 && R8>0) //If Camera 2 sees Target 8:
    {
        iH++;
        H = 0 + (180/3.14159)*asin((yR-y8t)/R8) - (P[2][8] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }



// CHECK CAMERA 3 SEES TARGETS 1 to 8
    if (P[3][1]>=0 && R1>0) //If Camera 3 sees Target 1:
    {
        iH++;
        H = +60 - (180/3.14159)*asin((yR-y1t)/R1) - (P[3][1] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[3][2]>=0 && R2>0) //If Camera 3 sees Target 2:
    {
        iH++;
        H = +60 + (180/3.14159)*asin((y2t-yR)/R2) - (P[3][2] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[3][3]>=0 && R3>0) //If Camera 3 sees Target 3:
    {
        iH++;
        H = -120 - (180/3.14159)*asin((y3t-yR)/R3) - (P[3][3] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[3][4]>=0 && R4>0) //If Camera 3 sees Target 4:
    {
        iH++;
        H = -120 + (180/3.14159)*asin((yR-y4t)/R4) - (P[3][4] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[3][5]>=0 && R5>0) //If Camera 3 sees Target 5:
    {
        iH++;
        H = +60 - (180/3.14159)*asin((yR-y5t)/R5) - (P[3][5] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[3][6]>=0 && R6>0) //If Camera 3 sees Target 6:
    {
        iH++;
        H = +60 + (180/3.14159)*asin((y6t-yR)/R6) - (P[3][6] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[3][7]>=0 && R7>0) //If Camera 3 sees Target 7:
    {
        iH++;
        H = -120 - (180/3.14159)*asin((y7t-yR)/R7) - (P[3][7] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }
    if (P[3][8]>=0 && R8>0) //If Camera 3 sees Target 8:
    {
        iH++;
        H = -120 + (180/3.14159)*asin((yR-y8t)/R8) - (P[3][8] - 640)*120./1280;
        if (H<0)
            H=H+360;
        if (H<0)
            H=H+360;
        Hsum=Hsum+H;
        ofs<<H<<endl;
    }


// COMPUTE THE AVERAGE OF ALL SOLUTIONS FOUND
    H=Hsum/iH;
    ofs<<"The average value of H is: "<<H<<endl;

}




