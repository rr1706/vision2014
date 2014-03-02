#-------------------------------------------------
#
# Project created by QtCreator 2012-01-28T14:32:54
#
#-------------------------------------------------

QT       += core

QT       -= gui

QT       += network

TARGET = Depth
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += \
    solutionlog.cpp \
    xyh.cpp \
    util.cpp \
    main.cpp \
    depthlogger.cpp \
    depthtools.cpp \
    imagewriter.cpp \
    target.cpp \
    ball.cpp \
    robot.cpp \
    ../lib/Webcam.cpp \
    ../lib/CameraFrame.cpp \
    ../lib/libcam.cpp \
    udpserver.cpp

LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui #`pkg-config --libs opencv`
QMAKE_CXXFLAGS += -std=c++0x -I/usr/include/ni

HEADERS += \
    util.hpp \
    solutionlog.hpp \
    xyh.hpp \
    depthlogger.h \
    depthtools.h \
    imagewriter.h \
    data.hpp \
    detection.hpp \
    config.hpp \
    ../lib/Webcam.hpp \
    ../lib/CameraFrame.hpp \
    ../lib/libcam.h \
    udpserver.h
