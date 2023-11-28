# Chest X-ray Image Classifier

## Description

A chest x-ray image classifier written in C++. Chest x-ray images were collected from a Kaggle  
datase[^1].
TensorFlow was used for training and classification itself. The model was trained based on  
[training images](training/preprocessed_images). A frozen graph of the trained model was then
produced and loaded into OpenCV for classification.



### Dependencies

* OpenCV 4.8.1
* TensorFlow 2.14

[^1]: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
