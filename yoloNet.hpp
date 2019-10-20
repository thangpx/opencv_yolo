#include <iostream>
#include <fstream>
// opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;

typedef struct yoloObject_t
{
    Rect boundingBox;
    String classId;
    float confidence;
} yoloObject_t;


class yoloNet
{
private:
    /* data */
    Net net;
    float confidenceThreshold;
    std::vector<String> classes;
    std::vector<String> net_outputNames;

    int width, height;
    std::vector<yoloObject_t> objects;
public:
    yoloNet(const String weightsPath, const String configPath, const String classesPath, 
            const int width = 608, const int height = 608, const float confidence = 0.5);
    ~yoloNet();

    void runOnFrame(Mat img);
    std::vector<yoloObject_t> getOutputObjects();
};
