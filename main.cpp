#include "yoloNet.hpp"

void drawBoundingBox(Mat &img, const String className, float confidence, Rect box) {
    // Draw rectangle
    int top = box.y - box.height/2;
    int left = box.x - box.width/2;
    rectangle(img, Rect(left, top, box.width, box.height), Scalar(0,255,0), 2);
    // Create the label text
    String labelTxt = format("%.2f",confidence);
    labelTxt = className + ":" + labelTxt;
    // Draw the label text on the image
    int baseline;
    Size labelSize = getTextSize(labelTxt, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    top = max(top, labelSize.height);
    rectangle(img, Point(left,top - labelSize.height), Point(left + labelSize.width, top + baseline), Scalar(0,0,0), FILLED);
    putText(img, labelTxt, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
}

int main(int argc, char const *argv[])
{
    Mat image;
    image = imread("street.jpg");
    // Create the YOLO network
    yoloNet yolo = yoloNet("yolov3.weights","yolov3.cfg","coco.names",608,608, 0.5);
    // run the network
    yolo.runOnFrame(image);
    // get the output
    std::vector<yoloObject_t> objects;
    objects = yolo.getOutputObjects();
    for(int i = 0; i < objects.size(); i++) {
        std::cout << objects[i].classId << ": " << objects[i].confidence << std::endl;
        drawBoundingBox(image, objects[i].classId, objects[i].confidence, objects[i].boundingBox);
    }
    // Show the image after prediction
    imshow("Output of prediction", image);
    waitKey(0);
    return 0;
}
