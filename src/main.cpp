#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <QtNetwork>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include "util.hpp"
#include "solutionlog.hpp"
#include "xyh.hpp"
#include "depthlogger.h"
#include "imagewriter.h"
#include "data.hpp"
#include "config.hpp"
#include "udpserver.h"
#include "../lib/Webcam.hpp"
#include "../lib/libcam.h"

const float BUMPER_HEIGHT = 5;

// OpenCV Namespace
using namespace cv;
using namespace std;

// for sprintf (grr)
char str[255];

// keys
const char KEY_QUIT = 27;
const char KEY_SAVE = 'w';
const char KEY_SPEED = ' ';

// config
ProcessingMode procMode = SA;
const InputSource mode = V4L2;
int cameraId = 2;
TrackMode tracking = TARGET;
const string videoPath = "Y400cmX646cm.avi";
// displayImage replaced with WindowMode::NONE
const TeamColor color = RED;
const bool doUdp = true;
const QHostAddress udpRecipient(arrayToIP("10.17.6.2")); //"192.168.17.17"/*"10.17.6.5"*/)); //(0x0A110602)
const short udpPort = 80;
QUdpSocket udpSocket;
const bool saveImages = true;
const double imageInterval = 1.0; // seconds

static const bool USE_POSE = false;
static const bool STITCH_IMAGES = false;

// Values for threshold IR
int gray_min = 210;
int gray_max = 255;

bool SAVE_IMAGES = true;
bool SAVE_LOGS = true;

ThresholdDataHSV ballThreshR = {115, 150, 116, 255, 100, 255};

// Values for threshold ball track
unsigned char ballHueMin = color == RED ? 115 : 31;
unsigned char ballHueMax = color == RED ? 150 : 128;
unsigned char ballSatMin = color == RED ? 116 : 92;
unsigned char ballSatMax = color == RED ? 255 : 202;
unsigned char ballValMin = color == RED ? 100 : 0;
unsigned char ballValMax = color == RED ? 255 : 158;
unsigned int ballSidesMin = 5; // for a circle
unsigned int ballMinArea = 250;
double ballRatioMin = 0.4;
double ballRatioMax = 0.9;

// for approxpolydp
int accuracy = 2; //maximum distance between the original curve and its approximation
int contourMinArea = 100;
float Tan_FOV_Y_Half = 1.46;

const int kern_mat0[] = {1,0,1,
                        0,1,0,
                        1,0,1};

const int kern_mat1[] = {1,1,1,
                        0,0,0,
                        1,1,1};

const int kern_mat2[] = {0,0,0,
                        0,0,0,
                        1,1,1};

Mat kernel0 = getStructuringElement(*kern_mat0, Size(3,3), Point(-1,-1));
Mat kernel1 = getStructuringElement(*kern_mat1, Size(3,3), Point(-1,-1));
Mat kernel2 = getStructuringElement(*kern_mat2, Size(3,3), Point(-1,-1));
Size winSize = Size( 5, 5 );
Size zeroZone = Size( -1, -1 );
TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

auto startTime = std::chrono::high_resolution_clock::now();
auto lastImageWrite = std::chrono::high_resolution_clock::now();

int imageWriteIndex = 0;
int dilations = 1;
WindowMode::WindowMode displayMode = WindowMode::FINAL;

char dirname[255];
time_t rawtime;
struct tm* timeinfo;
string windowName = "2014";
Thresh::Part currentThreshold = Thresh::HUE_MIN;
const int CHANGE_THRESH = 5;
const int CHANGE_AREA = 20;
const int CHANGE_ACCURACY = 1;
const int CHANGE_DILATE = 1;

const unsigned int TARGET_POINTS = 4;
const int CAMERA_OFFSET[CAMERA_COUNT] = {60, 180, 300};
int robotRotation = 0; // degrees, from alliance wall

const int STATIC_TARGET_IGNORE_THRESHOLD = 0; // inches

/**
  * List of coordinates in the world per target.
  * The vector is in the format of:
  * worldCoords[target] -> vector of points for the corners
  */
const vector<vector<Point3d> > worldCoords = {
    { // R1 R[0]
      {8.86,71.925,-5.9709},
      {32.86,71.925,-5.9709},
      {8.86,67.925,-5.9709},
      {32.86,67.925,-5.9709}
    },
    { // R2 R[1]
      {37.61,68.625,-0.0019},
      {41.61,68.625,-0.0019},
      {37.61,36.625,-0.0019},
      {41.61,36.625,-0.0019}
    },
    { // R3 R[2]
      {263.11,71.925,-5.9709},
      {287.11,71.925,-5.9709},
      {263.11,67.925,-5.9709},
      {287.11,67.925,-5.9709}
    },
    { // R4 R[3]
      {254.36,68.625,-0.0019},
      {258.36,68.625,-0.0019},
      {254.36,36.625,-0.0019},
      {258.36,36.625,-0.0019}
    },
    { // R5 R[4]
      {8.86,71.925,656.071},
      {32.86,71.925,656.071},
      {8.86,67.925,656.071},
      {32.86,67.925,656.071}
    },
    { // R6 R[5]
      {37.61,68.625,650.102},
      {41.61,68.625,650.102},
      {37.61,36.625,650.102},
      {41.61,36.625,650.102}
    },
    { // R7 R[6]
      {263.11,71.925,656.071},
      {287.11,71.925,656.071},
      {263.11,67.925,656.071},
      {287.11,67.925,656.071}
    },
    { // R8 R[7]
      {254.36,68.625,650.102},
      {258.36,68.625,650.102},
      {254.36,36.625,650.102},
      {258.36,36.625,650.102}
    }
};

const Mat distCoeffs = (Mat_<float>(5,1) << -1.3694165419404972e-01, 2.0525879204942091e-01, 0, 0, -1.3750202297193695e-01);
const Mat cameraMatrix = (Mat_<float>(3, 3) << 4.3636519036896868e+02 ,0,6.3950000000000000e+02, 0, 4.3636519036896868e+02, 3.5950000000000000e+02, 0,0, 1);
Mat rotation_vector, translation_vector, rotation_matrix, inverted_rotation_matrix, cw_translate;

int demo();
int sa();
int dlog();
void targetDetection(ThreadData &data);
void ballDetection(ThreadData &data);
void robotDetection(ThreadData &data);

void onSignal(int signum)
{
    if (signum == SIGABRT) {
    }
}

int main()
{
    switch (procMode) {
    case SA:
        return sa();
    case DEMO:
        return demo();
    case DEPTH:
        return dlog();
    }
}

int dlog()
{
    VideoCapture camera(CV_CAP_OPENNI_ASUS);
    if (!camera.isOpened()) return 1;
    DepthLogger log(camera);
    return log.start();
}

int demo()
{
    clock_t begin = clock();
    ImageWriter writer(true, 1.0, "demo");
    ThreadData data;
    CameraFrame camFrame;
    Webcam *v4l2Cam;
//    Camera cam("/dev/video1", 640, 480, 30);
    if (mode == CAMERA) {
        data.camera = VideoCapture(cameraId);
        if (!data.camera.isOpened()) {
            cerr << "Failed to open camera device id:" << cameraId << endl;
            return -1;
        }
        data.camera.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        data.camera.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    } else if (mode == VIDEO) {
        data.camera = VideoCapture(videoPath);
        if (!data.camera.isOpened()) {
            cerr << "Failed to open video path:" << videoPath << endl;
            return -1;
        }
    } else if (mode == V4L2) {
        sprintf(str, "/dev/video1706%d", cameraId);
        v4l2Cam = new Webcam(str, 1);
        v4l2Cam->SetResolution(640, 480);
        v4l2Cam->SetFPS(30);
        v4l2Cam->SetStreaming(true);
        v4l2Cam->SetControl(V4L2_CID_AUTO_WHITE_BALANCE, 0);
        v4l2Cam->SetControl(V4L2_CID_BRIGHTNESS, 0);
        v4l2Cam->SetControl(V4L2_CID_SHARPNESS, 4);
        v4l2Cam->SetControl(V4L2_CID_CONTRAST, 32);
        v4l2Cam->SetControl(V4L2_CID_SATURATION, 55);
        v4l2Cam->SetControl(V4L2_CID_HUE, 0);
        v4l2Cam->SetControl(V4L2_CID_WHITE_BALANCE_TEMPERATURE, 0);
        for (auto x : v4l2Cam->GetPixelFormats()) {
            unsigned char* fourcc = (unsigned char*) &x;
            printf("pix fmt %c%c%c%c\n", fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
        }
    } else if (mode == OCV_V4L2) {
        sprintf(str, "v4l2:///dev/video1706%d", cameraId);
        data.camera = VideoCapture(str);
        if (!data.camera.isOpened()) {
            cerr << "Failed to open camera device id:" << cameraId << endl;
            return -1;
        }
    }
    int currentSave = 0;
    int trackI = tracking;

    Mat img, inframe;

    if (mode == IMAGE) {
        inframe = imread("raw_img_0.png");
    }
    if (displayMode != WindowMode::NONE) {
        namedWindow(windowName, CV_WINDOW_NORMAL);
    }
    int frame = 0;
    data.targetLog.open("target.log", {"time", "frame", "image", "distance", "bound_height"});
    data.ballLog.open("ball.log", {"time", "img_x", "img_y", "rel_x", "rel_y", "distance", "rotation", "velocity", "heading"});
    while ( 1 )
    {
        clock_t loop_begin = clock();
        // Start timing a frame (FPS will be a measurement of the time it takes to process all the code for each frame)
        auto start = std::chrono::high_resolution_clock::now();

        if (mode == CAMERA || mode == VIDEO) {
            data.camera >> img;
        } else if (mode == IMAGE) {
            img = inframe.clone();
        } else if (mode == V4L2) {
            v4l2Cam->GetFrame(camFrame);
            img = camFrame.getMat();
        } else if (mode == OCV_V4L2) {
            data.camera >> img;
            cv::resize(img, img, resizeResolution);
        }
        if (FLIP_IMAGE && cameraId == FLIP_IMAGE_CAMERA) {
            cv::flip(img, img, -1);
        }

        //Break out of loop if esc is pressed
        switch (char key = waitKey(1)) {
        case KEY_QUIT:
            data.ballLog.flush();
            data.targetLog.flush();
            data.ballLog.close();
            data.targetLog.close();
            return 0;
            break;
        case KEY_SAVE:
            sprintf(str, "raw_img_%d.png", currentSave);
            imwrite(str, img);
            currentSave++;
            break;
        case KEY_SPEED:
            for (int gi = 0; gi < 20; gi++) {
                data.camera >> img;
            }
            break;
        case 'p':
            waitKey();
            break;
        case '-':
            switch (displayMode) {
            case WindowMode::THRESHOLD:
                switch (currentThreshold) {
                case Thresh::HUE_MIN:
                    ballHueMin -= CHANGE_THRESH;
                    break;
                case Thresh::HUE_MAX:
                    ballHueMax -= CHANGE_THRESH;
                    break;
                case Thresh::SAT_MIN:
                    ballSatMin -= CHANGE_THRESH;
                    break;
                case Thresh::SAT_MAX:
                    ballSatMax -= CHANGE_THRESH;
                    break;
                case Thresh::VAL_MIN:
                    ballValMin -= CHANGE_THRESH;
                    break;
                case Thresh::VAL_MAX:
                    ballValMax -= CHANGE_THRESH;
                    break;
                case Thresh::IR_MIN:
                    gray_min -= CHANGE_THRESH;
                    break;
                case Thresh::IR_MAX:
                    gray_max -= CHANGE_THRESH;
                    break;
                }
                break;
            case WindowMode::DILATE:
                dilations -= CHANGE_DILATE;
                break;
            case WindowMode::CONTOURS:
                contourMinArea -= CHANGE_AREA;
                break;
            case WindowMode::APPROXPOLY:
                accuracy -= CHANGE_ACCURACY;
                break;
            default:
                break;
            }
            break;
        case '+':
            switch (displayMode) {
            case WindowMode::THRESHOLD:
                switch (currentThreshold) {
                case Thresh::HUE_MIN:
                    ballHueMin += CHANGE_THRESH;
                    break;
                case Thresh::HUE_MAX:
                    ballHueMax += CHANGE_THRESH;
                    break;
                case Thresh::SAT_MIN:
                    ballSatMin += CHANGE_THRESH;
                    break;
                case Thresh::SAT_MAX:
                    ballSatMax += CHANGE_THRESH;
                    break;
                case Thresh::VAL_MIN:
                    ballValMin += CHANGE_THRESH;
                    break;
                case Thresh::VAL_MAX:
                    ballValMax += CHANGE_THRESH;
                    break;
                case Thresh::IR_MIN:
                    gray_min += CHANGE_THRESH;
                    break;
                case Thresh::IR_MAX:
                    gray_max += CHANGE_THRESH;
                    break;
                }
                break;
            case WindowMode::DILATE:
                dilations += CHANGE_DILATE;
                break;
            case WindowMode::CONTOURS:
                contourMinArea += CHANGE_AREA;
                break;
            case WindowMode::APPROXPOLY:
                accuracy += CHANGE_ACCURACY;
                break;
            default:
                break;
            }
            break;
        case 'h' :
            currentThreshold = Thresh::HUE_MIN;
            break;
        case 'H':
            currentThreshold = Thresh::HUE_MAX;
            break;
        case 's' :
            currentThreshold = Thresh::SAT_MIN;
            break;
        case 'S':
            currentThreshold = Thresh::SAT_MAX;
            break;
        case 'v' :
            currentThreshold = Thresh::VAL_MIN;
            break;
        case 'V':
            currentThreshold = Thresh::VAL_MAX;
            break;
        case 'i':
            currentThreshold = Thresh::IR_MIN;
            break;
        case 'I':
            currentThreshold = Thresh::IR_MAX;
            break;
        case ')':
            cameraId = 0;
            data.camera.open(0);
            break;
        case '!':
            cameraId = 1;
            data.camera.open(1);
            break;
        case '@':
            cameraId = 2;
            data.camera.open(2);
            break;
        case 'm':
            if (++trackI > ROBOT) trackI = 0;
            tracking = static_cast<TrackMode>(trackI);
            trackI = tracking;
            break;
        default:
            if (key >= '0' && key <= '9') {
                int ikey = key - '0';
                displayMode = static_cast<WindowMode::WindowMode>(ikey);
            }
        }
        writer.writeImage(img);
        static const int padding = 30;
        int topPortion = img.rows * (2.0/3);
        int bottomPortion = img.rows - topPortion;
        Mat topROI = img.clone(), bottomROI = img.clone();
        topROI.pop_back(bottomPortion + padding);
        copyMakeBorder(topROI, topROI, 0, bottomPortion + padding, 0, 0, BORDER_CONSTANT);
        for (int y = 0; y < topPortion + padding; y++) {
            for (int x = 0; x < img.cols; x++) {
                bottomROI.at<Vec3d>(Point(x, y)) = Vec3d(0, 0, 0);
            }
        }
        double timeSinceStart = static_cast<double>(loop_begin - begin) / CLOCKS_PER_SEC;
        switch (tracking) {
        case BALL:
            data.image = bottomROI;
            ballDetection(data);
            data.ballLog.log("frame", frame++).log("time", timeSinceStart).log("image", writer.imageIndex).flush();
            break;
        case TARGET:
            data.image = topROI;
            targetDetection(data);
            data.targetLog.log("frame", frame++).log("time", timeSinceStart).log("image", writer.imageIndex).flush();
            break;
        case ROBOT:
            data.image = bottomROI;
            robotDetection(data);
            break;
        }
        // Note: img is dirty after running these functions

        /// Stop timing and calculate FPS and Average FPS
        auto finish = std::chrono::high_resolution_clock::now();
        double seconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count()) / 1000000000.0;
        printf("Last frame: %.2f seconds.", seconds);
        fflush(stdout);
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

void runThread(ThreadData *data) {
    data->v4l2Cam->GetFrame(data->camFrame);
    data->image = data->camFrame.getMat();
//    data->camera >> data->image;
//    cv::pyrDown(data->image, data->image);
//    cv::pyrDown(data->image, data->image);
    if (resizeResolution != captureResolution)
        cv::resize(data->image, data->image, resizeResolution);
    if (data->id == 2) {
//        cv::flip(data->image, data->image, -1);
    }
    data->original = data->image.clone();
    static const int padding = 5;
    int topPortion = data->image.rows * (2.0/3);
    int bottomPortion = data->image.rows - topPortion;
    Mat topROI = data->image.clone(), bottomROI = data->image.clone();
    topROI.pop_back(bottomPortion + padding);
    copyMakeBorder(topROI, topROI, 0, bottomPortion + padding, 0, 0, BORDER_CONSTANT);
    for (int y = 0; y < 100; y++) {
        for (int x = 0; x < data->image.cols; x++) {
            topROI.at<Vec3d>(Point(x, y)) = Vec3d(0, 0, 0);
        }
    }
    for (int y = 0; y < topPortion + padding; y++) {
        for (int x = 0; x < data->image.cols; x++) {
            bottomROI.at<Vec3d>(Point(x, y)) = Vec3d(0, 0, 0);
        }
    }
    data->image = topROI;
    targetDetection(*data);
    data->targetDetect = data->image.clone();
    //    data->image = data->original.clone();
    data->image = bottomROI;
    ballDetection(*data);
}

ThreadData* threadData[CAMERA_COUNT];
std::thread* serverThread;
UDPServer* server;

void stopCameras() {
    for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
//            threadData[i]->camera.release();
        threadData[i]->v4l2Cam->SetStreaming(false);
        delete threadData[i]->v4l2Cam;
    }
    shutdown(server->socketFd, SHUT_RDWR);
    serverThread->join();
    delete server;
}

void saShutdown(int signum) {
    fprintf(stderr, "Abnormal program termination. Closing camera connections...\n");
    stopCameras();
    fprintf(stderr, "done.\n");
    exit(signum);
}

int sa()
{
    int cameraToDeviceTable[3] = {0, 1, 2};
    ImageWriter* writers[CAMERA_COUNT];
    string dirname = getDirnameNow();
    mkdir(dirname.c_str(), 0755);
    for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
        threadData[i] = new ThreadData;
        threadData[i]->id = cameraToDeviceTable[i];
        sprintf(str, "%s/cam_%d", dirname.c_str(), i);
        mkdir(str, 0755);
        sprintf(str, "cam_%d", i);
        if (SAVE_IMAGES) {
            writers[i] = new ImageWriter(false, 1.0, str, dirname);
        }
        if (SAVE_LOGS) {
            sprintf(str, "%s/cam_%d/ball.csv", dirname.c_str(), i);
            threadData[i]->ballLog.open(str, {"frame", "time", "image", "img_x", "img_y", "radius",
                                              "rel_x", "rel_y", "distance", "rotation"});
            sprintf(str, "%s/cam_%d/target.csv", dirname.c_str(), i);
            threadData[i]->targetLog.open(str, {"frame", "time", "image",
                                                "s0_dist", "s0_width", "s0_height", "s0_x", "s0_y",
                                                "s1_dist", "s1_width", "s1_height", "s1_x", "s1_y",
                                                "d0_dist", "d0_width", "d0_height", "d0_x", "d0_y",
                                                "d1_dist", "d1_width", "d1_height", "d1_x", "d1_y"
                                          });
        }
        sprintf(str, "/dev/video1706%d", cameraToDeviceTable[i]);
//        threadData[i]->camera.open(str);
        threadData[i]->v4l2Cam = new Webcam(str, 1);
        threadData[i]->v4l2Cam->SetResolution(captureResolution.width, captureResolution.height);
        threadData[i]->v4l2Cam->SetFPS(30);
        threadData[i]->v4l2Cam->SetStreaming(true);
        threadData[i]->v4l2Cam->SetControl(V4L2_CID_AUTO_WHITE_BALANCE, 0);
        threadData[i]->v4l2Cam->SetControl(V4L2_CID_BRIGHTNESS, 15);
        threadData[i]->v4l2Cam->SetControl(V4L2_CID_SHARPNESS, 4);
        threadData[i]->v4l2Cam->SetControl(V4L2_CID_CONTRAST, 32);
        threadData[i]->v4l2Cam->SetControl(V4L2_CID_SATURATION, 55);
        threadData[i]->v4l2Cam->SetControl(V4L2_CID_HUE, 0);
        threadData[i]->v4l2Cam->SetControl(V4L2_CID_WHITE_BALANCE_TEMPERATURE, 0);

    }
    server = new UDPServer(8888);
    serverThread = new std::thread(&UDPServer::listenLoop, server);
    signal(SIGTERM, saShutdown);
    signal(SIGINT, saShutdown);
    auto begin = std::chrono::high_resolution_clock::now();
    SolutionLog saLog("sa.csv", {"time", "frame_time", "case_0", "case_1", "case_2", "heading", "x", "y"});
    thread threads[CAMERA_COUNT];
    for (unsigned int frame = 0;; frame++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            threadData[i]->start = start;
            threads[i] = thread(runThread, threadData[i]);
        }
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            if (threads[i].joinable())
                threads[i].join();
        }
        double timeSinceStart = std::chrono::duration_cast<std::chrono::duration<double> >(start-begin).count();
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            if (SAVE_IMAGES) {
                writers[i]->writeImage(threadData[i]->original);
            }
            if (SAVE_LOGS) {
                threadData[i]->ballLog.log("frame", frame).log("time", timeSinceStart);
                threadData[i]->targetLog.log("frame", frame).log("time", timeSinceStart);
                if (SAVE_IMAGES) {
                    threadData[i]->ballLog.log("image", writers[i]->imageIndex).flush();
                    threadData[i]->targetLog.log("image", writers[i]->imageIndex).flush();
                }
                threadData[i]->ballLog.flush();
                threadData[i]->targetLog.flush();
            }
        }
        Point2i targetPair[CAMERA_COUNT][2] = {{Point2i(0, 0)}};
        if (threadData[2]->pairCase == LEFT) {
            targetPair[2][0] = Point2i(0, 4);
            if (threadData[0]->pairCase == LEFT) { // opposite side of screen, may be illegal
                if (threadData[2]->staticTargets[0].massCenter.x > 600 && threadData[0]->staticTargets[0].massCenter.x < 200) {
                    //ignore
                } else
                    targetPair[0][0] = Point2i(2, 6);
                if (threadData[1]->pairCase == LEFT) { // max 2 left pairs
                    fprintf(stderr, "\rPair error: 2L, 0L, 1L\n");
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 8);
                } else if (threadData[1]->pairCase == ALL) {
                    fprintf(stderr, "\rPair error: 2L, 0L, 1A\n");
                }
            } else if (threadData[0]->pairCase == RIGHT) {
                targetPair[0][0] = Point2i(1, 5);
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(2, 6);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 7);
                } else if (threadData[1]->pairCase == ALL) {
                    targetPair[1][0] = Point2i(2, 6);
                    targetPair[1][1] = Point2i(3, 7);
                }
            } else if (threadData[0]->pairCase == ALL) {
                targetPair[0][0] = Point2i(2, 6);
                targetPair[0][1] = Point2i(3, 7);
                if (threadData[1]->pairCase != NONE) {
                    fprintf(stderr, "\rPair error: 2L, 0A, 1!N\n");
                }
            } else {
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(2, 6);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 7);
                } else if (threadData[1]->pairCase == ALL) {
                    targetPair[1][0] = Point2i(2, 6);
                    targetPair[1][1] = Point2i(3, 7);
                }
            }
        } else if (threadData[2]->pairCase == RIGHT) {
            targetPair[2][0] = Point2i(1, 5);
            if (threadData[0]->pairCase == LEFT) {
                targetPair[0][0] = Point2i(2, 6);
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(0, 4);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 7);
                } else if (threadData[1]->pairCase == ALL) {
                    fprintf(stderr, "\rPair error: 2R, 0L, 1A\n");
                }
            } else if (threadData[0]->pairCase == RIGHT) {
                if (threadData[2]->staticTargets[0].massCenter.x > 600 && threadData[0]->staticTargets[0].massCenter.x < 200) {
                    //ignore
                } else
                    targetPair[0][0] = Point2i(3, 7);
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(0, 4);
                } else if (threadData[1]->pairCase == RIGHT || threadData[1]->pairCase == ALL) {
                    fprintf(stderr, "\rPair error: 2R, 0R, 1R|A\n");
                }
            } else if (threadData[0]->pairCase == ALL) {
                targetPair[0][0] = Point2i(2, 6);
                targetPair[0][1] = Point2i(3, 7);
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(0, 4);
                } else if (threadData[1]->pairCase == RIGHT || threadData[1]->pairCase == ALL) {
                    fprintf(stderr, "\rPair error: 2R, 0A, 1R|A\n");
                }
            } else {
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(2, 6);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 7);
                } else if (threadData[1]->pairCase == ALL) {
                    targetPair[1][0] = Point2i(2, 6);
                    targetPair[1][1] = Point2i(3, 7);
                }
            }
        } else if (threadData[2]->pairCase == ALL) {
            targetPair[2][0] = Point2i(0, 4);
            targetPair[2][1] = Point2i(1, 5);
            if (threadData[0]->pairCase == LEFT) {
                targetPair[0][0] = Point2i(2, 6);
                if (threadData[1]->pairCase == LEFT) {
                    fprintf(stderr, "\rPair error: 2A, 0L, 1L\n");
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 7);
                } else if (threadData[1]->pairCase == ALL) {
                    fprintf(stderr, "\rPair error: 2A, 0L, 1A\n");
                }
            } else if (threadData[0]->pairCase == RIGHT) {
                bool ignore = false;
                if (threadData[2]->staticTargets.size() == 2) {
                    if (threadData[2]->staticTargets[1].massCenter.x > 600 && threadData[0]->staticTargets[0].massCenter.x < 200) {
                        ignore = true;
                    }
                }
                if (threadData[2]->dynamicTargets.size() == 2) {
                    if (threadData[2]->dynamicTargets[1].massCenter.x > 600 && threadData[0]->staticTargets[0].massCenter.x < 200) {
                        ignore = true;
                    }
                }
                if (!ignore)
                    targetPair[0][0] = Point2i(3, 7);
                if (threadData[1]->pairCase != NONE) {
                    fprintf(stderr, "\rPair error: 2A. 0R, 1!N\n");
                }
            } else if (threadData[0]->pairCase == ALL) {
                // should not be possible, but idk
                targetPair[0][0] = Point2i(2, 6);
                targetPair[0][1] = Point2i(3, 7);
                if (threadData[1]->pairCase != NONE) {
                    fprintf(stderr, "\rPair error: 2A, 0A, 1!N\n");
                }
            } else {
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(2, 6);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 7);
                } else if (threadData[1]->pairCase == ALL) {
                    // should not be possible, but idk
                    targetPair[1][0] = Point2i(2, 6);
                    targetPair[1][1] = Point2i(3, 7);
                }
            }
        } else {
            if (threadData[0]->pairCase == LEFT) {
                targetPair[0][0] = Point2i(0, 4);
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(2, 6);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(1, 5);
                } else if (threadData[1]->pairCase == ALL) {
                    targetPair[1][0] = Point2i(2, 6);
                    targetPair[1][1] = Point2i(1, 5);
                }
            } else if (threadData[0]->pairCase == RIGHT) {
                targetPair[0][0] = Point2i(1, 5);
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(2, 6);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 7);
                } else if (threadData[1]->pairCase == ALL) {
                    targetPair[1][0] = Point2i(2, 6);
                    targetPair[1][1] = Point2i(3, 7);
                }
            } else if (threadData[0]->pairCase == ALL) {
                targetPair[0][0] = Point2i(0, 4);
                targetPair[0][1] = Point2i(1, 5);
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(2, 6);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(3, 7);
                } else if (threadData[1]->pairCase == ALL) {
                    targetPair[1][0] = Point2i(2, 6);
                    targetPair[1][1] = Point2i(3, 7);
                }
            } else {
                if (threadData[1]->pairCase == LEFT) {
                    targetPair[1][0] = Point2i(0, 4);
                } else if (threadData[1]->pairCase == RIGHT) {
                    targetPair[1][0] = Point2i(1, 5);
                } else if (threadData[1]->pairCase == ALL) {
                    targetPair[1][0] = Point2i(0, 4);
                    targetPair[1][1] = Point2i(1, 5);
                }
            }
        }
        int P[CAMERA_COUNT][TARGET_COUNT];
        for (unsigned int pi = 0; pi < CAMERA_COUNT; pi++) {
            for (unsigned int pj = 0; pj < TARGET_COUNT; pj++) {
                P[pi][pj] = -1;
            }
        }
        double robotDistance = 0, robotAngle = 0;
        if (threadData[0]->robotISA != -99
                && threadData[0]->robotISA <= -30 && threadData[2]->robotISA > 30) {
            // ensure we are overlapping
            double rightAngle = 150 - threadData[0]->robotISA;
            double leftAngle = 150 - threadData[2]->robotISA;
            // double centerAngle = 180 - rightAngle - leftAngle;
            // tan (left) * n = 13 * tan (right) - tan (right) * n    SOLVE FOR N
            double tanR = tan(rightAngle * (CV_PI / 180));
            double tanL = tan(leftAngle * (CV_PI / 180));
            // tanL * N = (13 * tanR) - (tanR * n)
            // tanR * N + tanL * N = 13 * tanR
            // (tanR + tanL) * N = 13 * tanR
            double N = (13 * tanR) / (tanR + tanL);
            double distance1 = tanR * N;
            double distance2 = tanL * (13 - N);
            robotDistance = (distance1 + distance2) / 2;
            double robotAngle1 = threadData[0]->robotISA + 30;
            double robotAngle2 = threadData[2]->robotISA - 30;
            robotAngle = (robotAngle1 + robotAngle2) / 2;
        }
        std::vector<cv::Point3f> dataWorldCoords;
        double R[8] = {0};
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            ThreadData *data = threadData[i];
            sprintf(str, "case_%d", i);
            saLog.log(str, data->pairCase);

            if (data->pairCase == LEFT && data->staticTargets.size() > 0
                    && data->dynamicTargets.size() > 0) {
//                if (abs(data->staticTargets[0].realDistance - data->dynamicTargets[0].realDistance) > STATIC_TARGET_IGNORE_THRESHOLD)
                    R[targetPair[i][0].y] = data->staticTargets[0].realDistance;
                R[targetPair[i][0].x] = data->dynamicTargets[0].realDistance;
                P[i][targetPair[i][0].y] = data->staticTargets[0].massCenter.x;
                P[i][targetPair[i][0].x] = data->dynamicTargets[0].massCenter.x;
                // do dynamic before static, x is for dynamic
                for (int j = 0; j < 4; j++) {
                    dataWorldCoords.push_back(worldCoords[targetPair[i][0].x][j]);
                }
                for (int j = 0; j < 4; j++) {
                    dataWorldCoords.push_back(worldCoords[targetPair[i][0].y][j]);
                }
            } else if (data->pairCase == RIGHT && data->staticTargets.size() > 0
                       && data->dynamicTargets.size() > 0) {
//                if (abs(data->staticTargets[0].realDistance - data->dynamicTargets[0].realDistance) > STATIC_TARGET_IGNORE_THRESHOLD)
                    R[targetPair[i][0].y] = data->staticTargets[0].realDistance;
                R[targetPair[i][0].x] = data->dynamicTargets[0].realDistance;
                P[i][targetPair[i][0].y] = data->staticTargets[0].massCenter.x;
                P[i][targetPair[i][0].x] = data->dynamicTargets[0].massCenter.x;
                for (int j = 0; j < 4; j++) {
                    dataWorldCoords.push_back(worldCoords[targetPair[i][0].x][j]);
                }
                for (int j = 0; j < 4; j++) {
                    dataWorldCoords.push_back(worldCoords[targetPair[i][0].y][j]);
                }
            } else if (data->pairCase == ALL && data->staticTargets.size() > 1
                       && data->dynamicTargets.size() > 1) {
//                if (abs(data->staticTargets[0].realDistance - data->dynamicTargets[0].realDistance) > STATIC_TARGET_IGNORE_THRESHOLD)
                    R[targetPair[i][0].x] = data->staticTargets[0].realDistance;
                R[targetPair[i][0].y] = data->dynamicTargets[0].realDistance;
//                if (abs(data->staticTargets[1].realDistance - data->dynamicTargets[1].realDistance) > STATIC_TARGET_IGNORE_THRESHOLD)
                    R[targetPair[i][1].x] = data->staticTargets[1].realDistance;
                R[targetPair[i][1].y] = data->dynamicTargets[1].realDistance;
                P[i][targetPair[i][0].x] = data->staticTargets[0].massCenter.x;
                P[i][targetPair[i][0].y] = data->dynamicTargets[0].massCenter.x;
                P[i][targetPair[i][1].x] = data->staticTargets[1].massCenter.x;
                P[i][targetPair[i][1].y] = data->dynamicTargets[1].massCenter.x;
                for (int j = 0; j < 4; j++) {
                    dataWorldCoords.push_back(worldCoords[targetPair[i][0].x][j]);
                }
                for (int j = 0; j < 4; j++) {
                    dataWorldCoords.push_back(worldCoords[targetPair[i][0].y][j]);
                }
                for (int j = 0; j < 4; j++) {
                    dataWorldCoords.push_back(worldCoords[targetPair[i][1].x][j]);
                }
                for (int j = 0; j < 4; j++) {
                    dataWorldCoords.push_back(worldCoords[targetPair[i][1].y][j]);
                }
            }
        }
        for (int i = 0; i < CAMERA_COUNT; i++) {
            for (int j = 0; j < TARGET_COUNT; j++) {
                if (P[i][j] != -1) {
                    P[i][j] *= 1.5;
                }
            }
        }
        for (int i = 0; i < 8; i++) {
            if (R[i] < 0) R[i] = 0;
        }
        double xPos, yPos, heading;
        FindXYH(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], P, xPos, yPos, heading);
        if (xPos == 0 )
        {
            xPos = -1;
            yPos = -1;
        }
//        printf("\rFindXYH(%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f): %.2f, %.2f, %.2f\n",
//               R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], xPos, yPos, heading);
        if (USE_POSE) {
            cv::Mat rvec(1,3,cv::DataType<double>::type);
            cv::Mat tvec(1,3,cv::DataType<double>::type);
            cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
            double theta = 0, psi = 0, phi = 0;
            Point3d euler(theta, psi, phi);
            //cv::solvePnP(dataWorldCoords, localCorners, cameraMatrix, distCoeffs, rvec, tvec);
            cv::Rodrigues(rvec,rotationMatrix);
            //        Rodrigues (rvec, rotation_matrix);
            //        Rodrigues (rotation_matrix.t (), camera_rotation_vector);
            Mat t = tvec.t ();
            //        camera_translation_vector = -camera_rotation_vector * t;
        }
        unsigned int largestBall = 0;
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            if (threadData[i]->ballArea > threadData[largestBall]->ballArea) {
                largestBall = i;
            }
        }
        int whichGoalHot = 0;
        if (threadData[2]->pairCase == LEFT && threadData[0]->pairCase == RIGHT) {
            whichGoalHot = 0;
        } else if (threadData[2]->pairCase == LEFT ){//|| threadData[2]->pairCase == LEFT) {
            whichGoalHot = 1;
        } else if (threadData[0]->pairCase == RIGHT ){//||threadData[2]->pairCase == RIGHT) {
            whichGoalHot = 2;
        }
        const char* hotGoal;
        switch (whichGoalHot) {
        case 1:
            hotGoal = "left";
            break;
        case 2:
            hotGoal = "right";
            break;
        default:
            hotGoal = "unk";
        }

        printf("\rx %.2f y %.2f h%.2f hot %s bd %.2f ba %.2f bv %.2f bh %.2f rd %.2f ra %.2f\n",
               xPos, yPos, heading, hotGoal, threadData[largestBall]->distanceToBall,
               threadData[largestBall]->angleToBall, threadData[largestBall]->ballVelocity,
               threadData[largestBall]->ballHeading, robotDistance, robotAngle);

        if (doUdp) {
            QByteArray datagram = QByteArray::number(xPos) + " "
                    + QByteArray::number(yPos) + " "
                    + QByteArray::number(heading) + " "
                    + QByteArray::number(whichGoalHot) + " "
                    + QByteArray::number(threadData[largestBall]->distanceToBall) + " "
                    + QByteArray::number(threadData[largestBall]->angleToBall) + " "
                    + QByteArray::number(threadData[largestBall]->ballVelocity) + " "
                    + QByteArray::number(threadData[largestBall]->ballHeading) + " "
                    + QByteArray::number(robotDistance) + " "
                    + QByteArray::number(robotAngle);
            udpSocket.writeDatagram(datagram.data(), datagram.size(), udpRecipient, udpPort);
        }
        // stitching is broken currently, TODO test on odroid
        if (STITCH_IMAGES) {
            Mat pano;
            vector<Mat> imgs;
            Stitcher stitcher = Stitcher::createDefault();
            for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
                imgs.push_back(threadData[i]->original);
            }
            Stitcher::Status status = stitcher.stitch(imgs, pano);
            if (status != Stitcher::OK) {
                cerr << "Error stitching - Code: " << int(status) << endl;
                imshow(windowName, threadData[0]->original);
            } else {
                Window::print("Ratchet Rockers 1706", pano, Point(pano.cols - 200, 15));
                sprintf(str, "xPos %.2f yPos %.2f heading %.2f", xPos, yPos, heading);
                Window::print(string(str), pano, Point(5, 15));
                imshow(windowName, pano);
            }
        } else if (false && displayMode != WindowMode::NONE) {
            imshow("cam-zero", threadData[0]->dst);
            imshow("cam-one", threadData[1]->dst);
            imshow("cma-two", threadData[2]->dst);
        }
        for (unsigned int i = 0; i < CAMERA_COUNT; i++) {
            threadData[i]->targets.clear();
            threadData[i]->staticTargets.clear();
            threadData[i]->dynamicTargets.clear();
        }

        auto finish = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(finish-start).count();
        double tSnSt = std::chrono::duration_cast<std::chrono::duration<double> >(finish-begin).count();
        saLog.log("time", tSnSt).log("frame_time", seconds).log("heading", heading).log("x", xPos).log("y", yPos).flush();
        printf("Last frame: %.2f seconds", seconds);
        fflush(stdout);
        switch (char key = waitKey(10)) {
        case 27:
            stopCameras();
            return key - 27;
        }
    }
    return 0;
}
