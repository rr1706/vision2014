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
    main.cpp

LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d
QMAKE_CXXFLAGS += -std=c++0x

HEADERS += \
    util.hpp \
    solutionlog.hpp \
    xyh.hpp
