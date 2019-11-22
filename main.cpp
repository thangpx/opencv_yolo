#include "yoloNet.hpp"
#include <Windows.h>

using namespace cv;
using namespace dnn;

void consoleMove(int shiftX, int shiftY) {
	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(hConsole, &coninfo);
	coninfo.dwCursorPosition.Y += shiftY;    // move up one line
	coninfo.dwCursorPosition.X += shiftX;    // move to the right the length of the word
	SetConsoleCursorPosition(hConsole, coninfo.dwCursorPosition);
}

void drawBoundingBox(Mat &img, int id, float confidence, Rect box) {
    // Draw rectangle
    int top = box.y - box.height/2;
    int left = box.x - box.width/2;
    rectangle(img, Rect(left, top, box.width, box.height), Scalar(0,255,0), 3);
    // Create the label text
    String labelTxt = format("%.2f",confidence);
	String idTxt = format("[%d] ", id);
    labelTxt = idTxt + labelTxt;
    // Draw the label text on the image
    int baseline;
    Size labelSize = getTextSize(labelTxt, FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
    top = max(top, labelSize.height);
    rectangle(img, Point(left,top - labelSize.height), Point(left + labelSize.width, top + baseline), Scalar(0,0,0), FILLED);
    putText(img, labelTxt, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 3);
}

std::string keys =
"{ help  h     | | Print help message. }"
"{ input i     | | Path to input video file.}";

int main(int argc, char const *argv[])
{
	CommandLineParser parser(argc, argv, keys);
	const std::string imgFile = parser.get<String>("i");
    Mat frame;
    // Read video parameter
    VideoCapture cap(imgFile);
    // resolution for output video
    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CV_CAP_PROP_FPS);
    // create object for the output video
    VideoWriter vWrite("video_yolo.avi",CV_FOURCC('M','J','P','G'),fps,Size(frame_width,frame_height));
    // Create the YOLO network
    yoloNet yolo = yoloNet("yolov3-tiny_face.weights","yolov3-tiny_face.cfg","coco.names",416,416, 0.3);

	int previous_nObject = 0;
    while (1)
    {
        // load frame
        cap >> frame;
        if(frame.empty()) break;

        // run the network
        yolo.runOnFrame(frame);
        // get the output
        std::vector<yoloObject_t> objects;
        objects = yolo.getOutputObjects();

        putText(frame, std::to_string(objects.size()), Point(frame.cols / 5, 100), FONT_HERSHEY_SIMPLEX, 4, Scalar(255,255,255), 5);

		consoleMove(0, -previous_nObject);
        for(int i = 0; i < objects.size(); i++) {
			printf("%3d. [", i);
			int nPercent = (int)((objects[i].confidence + 0.05) * 10);
			for (int j = 1; j <= 10; j++) {
				if (j <= nPercent) printf("=");
				else printf(" ");
			}
			std::cout << "] " << objects[i].confidence << std::endl;
            drawBoundingBox(frame, i, objects[i].confidence, objects[i].boundingBox);
        }
		if (objects.size() < previous_nObject) {
			for (int i = objects.size(); i < previous_nObject; i++) {
				for (int j = 0; j < 35; j++) std::cout << ' ';
				std::cout << std::endl;
			}
			consoleMove(0, -(previous_nObject - objects.size()));
		}
		previous_nObject = objects.size();
        // write
        vWrite.write(frame);

        // show the frame
		namedWindow("Face detection reult", WINDOW_NORMAL);
		resizeWindow("Face detection reult", frame.cols / 2, frame.rows / 2);
        imshow("Face detection reult",frame);

        // press ESC to exit
        char c = (char)waitKey(1);
        if(c == 27) break;
    }

    // release
    cap.release();
    vWrite.release();
    
    destroyAllWindows();
    return 0;
}
