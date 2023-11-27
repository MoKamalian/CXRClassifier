/**
 *
 * @author: amir kamalian
 * @date:   28 sep 2023
 *
 * @reference: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
 *
 * */


#include <iostream>
#include <regex>
#include <opencv2/opencv.hpp>
#include "../inc/Colors.hpp"


using std::cout, std::endl;
using namespace cv;

/** @brief command line argument parser */
bool parse_command_line(int argc, char** args);

int main(int argc, char** argv) {

    if(!parse_command_line(argc, argv)) {
       cout << BOLDRED << "[error invalid arguments provided] " << endl;
       cout << "[image file provided must be jpeg]" << RESET << endl;
       exit(0);
    }

    /* read in image to be classified */
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    dnn::Net net = dnn::readNetFromTensorflow("../training/frozen_graph/cxr_classifier.pb");

    try {
        if (net.empty()) {
            std::cerr << "Error: Failed to load the TensorFlow model." << endl;
            return -1;
        }
    } catch (const cv::Exception& ex) {
        std::cerr << "OpenCV Exception: " << ex.what() << endl;
        return -1;
    } catch (const std::exception& ex) {
        std::cerr << "Standard Exception: " << ex.what() << endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown Exception." << endl;
        return -1;
    }


    return 0;
}

/** @brief command line argument parser */
bool parse_command_line(int argc, char** argv) {
    if(argc != 2) {
        cout << BOLDRED << "[error no arguments provided]" << RESET << endl;
        return false;
    }

    static const std::regex expression = std::regex(R"(\b(\.jpeg))");
    return std::regex_search(argv[1], expression); // returns true if extension match is found
}







