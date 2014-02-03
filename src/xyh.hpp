#ifndef XYH_HPP
#define XYH_HPP

void FindXYH(double, double, double, double, double, double, double, double, int P[4][9], double&, double&, double&);
void FindXY(double, double, double, double, double, double, double, double);
void FindPlusMinusSolutionInFieldCooridinates(double, double, double, double, double, double);
void SelectSolutionForAdjacentCorners();
void SelectSolutionForOppositeCorners(double, double, double, double);
void FindHeading(double, double, double, double, double, double, double, double, int P[4][9]);

#endif // XYH_HPP
