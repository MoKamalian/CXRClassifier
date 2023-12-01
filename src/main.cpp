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
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../inc/Colors.hpp"


using std::cout, std::endl, std::string;
using namespace cv;
using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

/** @brief command line argument parser */
bool parse_command_line(int argc, char** args) noexcept;

/** @brief for printing of the predicted results of the model */
void print_prediction(cv::Mat i) noexcept;

int main(int argc, char** argv) {

    //TODO: REDO THE COMMAND LINE ARG VALIDATION TO VALIDATE FOR DIRECTORY ENTRIES WITH EXISTING IMAGES
    /*0
    if(!parse_command_line(argc, argv)) {
       cout << BOLDRED << "[error invalid arguments provided] " << endl;
       cout << "[image file provided must be jpeg]" << RESET << endl;
       exit(0);
    }*/

    string path1 = argv[1];
    string path2 = argv[2];

    /* model input */
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

    /* classification of a single test image */
    /* IMAGE SIZES ARE (25, 256) */

    /* loop through normal test set images */
    cout << BOLDMAGENTA << "++++++++++++++++++++NORMAL++++++++++++++++++++" << RESET << endl;
    for(auto& img : recursive_directory_iterator(path1)) {

        /* read in image to be classified */
        cv::Mat image = cv::imread(img.path(), cv::IMREAD_GRAYSCALE);
        cv::Mat img_blob = cv::dnn::blobFromImage(image, 1.0, Size(256, 256),
                                                  Scalar(127.5, 127.5, 127.5),
                                                  true, false);

        try {

            net.setInput(img_blob);
            cv::Mat prediction = net.forward();

            print_prediction(prediction);

        } catch (cv::Exception &e) {
            cout << e.what() << endl;
        }
    }

    cout << BOLDBLUE << "++++++++++++++++++++PNEUMONIA++++++++++++++++++++" << RESET << endl;

    /* loop through pneumonia test images */
    for(auto& img : recursive_directory_iterator(path2)) {

        /* read in image to be classified */
        cv::Mat image = cv::imread(img.path(), cv::IMREAD_GRAYSCALE);
        cv::Mat img_blob = cv::dnn::blobFromImage(image, 1.0, Size(256, 256),
                                                  Scalar(127.5, 127.5, 127.5),
                                                  true, false);

        try {

            net.setInput(img_blob);
            cv::Mat prediction = net.forward();

            print_prediction(prediction);

        } catch (cv::Exception &e) {
            cout << e.what() << endl;
        }
    }




    return 0;
}

/** @brief command line argument parser */
bool parse_command_line(int argc, char** argv) noexcept {
    if(argc != 2) {
        cout << BOLDRED << "[error no arguments provided]" << RESET << endl;
        return false;
    }

    static const std::regex expression = std::regex(R"(\b(\.jpeg))");
    return std::regex_search(argv[1], expression); // returns true if extension match is found
}

/** @brief for printing of the predicted results of the model */
void print_prediction(cv::Mat i) noexcept {
    if(i.at<int>(0) == 0) {
        cout << BOLDGREEN << "Prediction: " << i << RESET << endl;
    } else {
        cout << BOLDRED << "Prediction: " << i << RESET << endl;
    }
}





