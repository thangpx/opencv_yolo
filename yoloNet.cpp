
#include "yoloNet.hpp"

using namespace cv;
using namespace dnn;

yoloNet::yoloNet(const String weightsPath, const String configPath, const String classesPath, 
            const int width, const int height, const float confidence)
{
    
    this->confidenceThreshold = confidence;
    this->width = width;
    this->height = height;

    // load class names
    std::ifstream classes_ifs(classesPath.c_str());
    CV_Assert(classes_ifs.is_open());
    std::string line;

    while (std::getline(classes_ifs, line))
    {
        this->classes.push_back(line);
    }

    // create the network
    this->net = readNetFromDarknet(configPath, weightsPath);
    this->net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    this->net.setPreferableTarget(DNN_TARGET_CPU);

    // get the names of the unconnected output layers
    this->net_outputNames = this->net.getUnconnectedOutLayersNames();
}

yoloNet::~yoloNet()
{
}

void yoloNet::runOnFrame(Mat img) {
    // convert the image to 4-D blob
    Mat blob;
    blobFromImage(img, blob, 1/255.0, Size(this->width,this->height), Scalar(0,0,0), true, false);
    // set the network input
    net.setInput(blob);
    // run the network
    std::vector<Mat> netOuts;
    net.forward(netOuts, this->net_outputNames);

    // decide the classes for bounding boxes
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<Rect> boundingBoxes;

    for(int i = 0; i < netOuts.size(); i++) {
        float* data = (float*)netOuts[i].data;
        for(int j = 0; j < netOuts[i].rows; j++) {
            float objectnessPrediction = data[4];
            if(objectnessPrediction >= this->confidenceThreshold) {
                Mat classPredictions = netOuts[i].row(j).colRange(5,netOuts[i].cols);
                Point maxPoint;
                double maxVal;
                minMaxLoc(classPredictions, 0, &maxVal, 0, &maxPoint);

                int centerX = (int)(data[0] * img.cols);
                int centerY = (int)(data[1] * img.rows);
                int boxWidth = (int)(data[2] * img.cols);
                int boxHeight = (int)(data[3] * img.rows);

                confidences.push_back(maxVal);
                classIds.push_back(maxPoint.x);
                boundingBoxes.push_back(Rect(centerX, centerY, boxWidth, boxHeight));
            }
            data += netOuts[i].cols;
        }
    }
    // remove the bounding boxes indicate the same object using NMS
    std::vector<int> indices;
    NMSBoxes(boundingBoxes, confidences, this->confidenceThreshold, 0.3, indices);
    // save bounding boxes
    this->objects.resize(indices.size());
    for(int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        CV_Assert(classIds[idx] < this->classes.size());
		yoloObject_t object;
		object.boundingBox = boundingBoxes[idx];
		object.classId = this->classes[classIds[idx]];
		object.confidence = confidences[idx];
        
        this->objects[i] = object;
    }
}

std::vector<yoloObject_t> yoloNet::getOutputObjects() {
    return this->objects;
}