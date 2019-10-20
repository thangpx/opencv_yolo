cmake_minimum_required(VERSION 2.8)
project( opencv_yolo )
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories( ${OpenCV_INCLUDE_DIRS}
                    ./ 
                    )
add_executable( 
            # The name of the executable file
            opencv_yolo
            # Dependence source files
            main.cpp 
            yoloNet.cpp 
        )
target_link_libraries( opencv_yolo ${OpenCV_LIBS} )