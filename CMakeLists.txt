cmake_minimum_required(VERSION 2.8)
project( opencv_yolo )
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
option(OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES 1)
set(OPENCV_OPENCL_DEVICE NVIDIA:GPU:)
option(WITH_CUDA "Enable CUDA support" ON)

include_directories( ${OpenCV_INCLUDE_DIRS}
                    ./ 
                    )
add_executable( 
            # The name of the executable file
            opencv_yolo
            # Dependence source files
            main.cpp 
            yoloNet.cpp
            yoloNet.hpp
        )
target_link_libraries( opencv_yolo ${OpenCV_LIBS} )