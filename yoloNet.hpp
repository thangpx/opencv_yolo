#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
// opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

typedef struct yoloObject_t
{
    cv::Rect boundingBox;
    cv::String classId;
    float confidence;
} yoloObject_t;


class yoloNet
{
private:
    /* data */
	cv::dnn::Net net;
    float confidenceThreshold;
    std::vector<cv::String> classes;
    std::vector<cv::String> net_outputNames;

    int width, height;
    std::vector<yoloObject_t> objects;
public:
    yoloNet(const cv::String weightsPath, const cv::String configPath, const cv::String classesPath,
            const int width = 608, const int height = 608, const float confidence = 0.5);
    ~yoloNet();

    void runOnFrame(cv::Mat img);
    std::vector<yoloObject_t> getOutputObjects();
};
