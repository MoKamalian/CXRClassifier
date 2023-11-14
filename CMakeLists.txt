cmake_minimum_required(VERSION 3.24)
project(CXRClassifier)

set(CMAKE_CXX_STANDARD 23)

add_executable(CXRClassifier src/main.cpp)

# OpenCV Linked header and library
include_directories(/opt/homebrew/Cellar/opencv/4.8.1_2/include/opencv4/)
target_link_directories(CXRClassifier PUBLIC /opt/homebrew/Cellar/opencv/4.8._2/lib/)
target_link_libraries(CXRClassifier /opt/homebrew/Cellar/opencv/4.8.1_2/lib/libopencv_core.4.8.1.dylib
                                    /opt/homebrew/Cellar/opencv/4.8.1_2/lib/libopencv_imgproc.4.8.1.dylib
                                    /opt/homebrew/Cellar/opencv/4.8.1_2/lib/libopencv_img_hash.4.8.1.dylib
                                    /opt/homebrew/Cellar/opencv/4.8.1_2/lib/libopencv_imgcodecs.4.8.1.dylib
                                    /opt/homebrew/Cellar/opencv/4.8.1_2/lib/libopencv_intensity_transform.4.8.1.dylib
        )


