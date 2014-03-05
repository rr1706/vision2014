2014 UDP Interface
==================


Input: floating point numbers in a space-separated string. Note the hot goal parameter is an integer.

           "212.19 123.64 234.3 1 7.34 13.7 0.42 235.2 4.87 238"
    xPos <----|       |      |  |   |    |    |    |    |    |
    yPos <------------|      |  |   |    |    |    |    |    |
    heading <----------------|  |   |    |    |    |    |    |
    hot goal <------------------|   |    |    |    |    |    |
    distance to ball <--------------|    |    |    |    |    |
    angle to ball <----------------------|    |    |    |    |
    ball velocity <---------------------------|    |    |    |
    ball heading <---------------------------------|    |    |
    robot distance <------------------------------------|    |
    angle to robot <-----------------------------------------|


xPos
----

X position on the field, calculated by FindXYH. Refer to Bob's field & targets diagram for positioning information.

This value is in inches.

yPos
----

Y position on the field, calculated by FindXYH. Refer to Bob's field & targets diagram for positioning information.

This value is in inches.

heading
-------

Heading of the robot, based on the starting position on the field. This value is in degrees.

hot goal
--------

Integer value.

Value is 0 if we can't figure it out.
Value is 1 if the target is left.
Value is 2 if the target is right.

distance to ball
----------------

Distance to our team's ball.

This value is in meters.


angle to ball
-------------

The angle that our robot is rotated away from our team's ball.

This value is in degrees.

ball velocity
-------------

The speed the ball is moving based on the last frame the robot saw.

This value is in meters per second.

Caution: Here be demons.

ball heading
------------

The heading the ball is moving from its last position.

This value is in degrees.

robot distance
--------------

The distance our robot is away from the closest one it can see.

angle to robot
--------------

The angle our robot is rotated away from the closest one we can see.

