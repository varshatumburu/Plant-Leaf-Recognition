# Plant Leaf Recognition
Plant recognition has been a challenging study since the early last century. With the development of image processing and pattern recognition, plant recognition has become possible. Automatic plant species recognition with image processing has applications in weeds identification, species discovery, plant taxonomy, disease detection and even natural reserve park management.

## Task Description 
This project aims to classify leaves using traditional handcrafted features and features extracted from pre-trained deep convolutional neural networks (ConvNets). The input to our system is raw images from a dataset and the output is the label for each species. By the end of the project, we should be able to predict a plant species by taking in a leaf image from the user. 

## Approach 
Feature extraction plays the most important role in determining the accuracy and precision of a machine learning mechanism. This is stemmed from the fact that the architecture for machine learning essentially depends on the predetermined feature that is prompted into the network. There is no single feature extraction technique that can be considered as the best method since different problems require different approaches and different mechanisms. 

I have used the [Swedish Leaf Dataset](https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/) for my project due to the larger training images/ class factor and a constraint on resources. It provides images with only one leaf and a clean background, making it more distant from the real world scenario. However, I have introduced a mask which filters noisy images, thereby making the set of feasible inputs more expansive. Unlike other classifiers, the Convolutional Neural Network(CNN) extracts and identifies the features concurrently; hence it is a faster recognition process. However, this classifier requires users to train numerous sets of data before it is considered to be competent enough for application. Therefore, a CNN that incorporates deep learning as its mechanism seems to be remarkably accurate in plant recognition.

## Deployment
Make sure to have python installed before going with these commands. To install all the required dependencies after cloning this repo:

``` python -m pip install -r requirements.txt ```

Run the below command to run the application on your localhost:
``` streamlit run app.py ```

You can find some working model images and a video demo in this [presentation](https://docs.google.com/presentation/d/1LFptaFhFhMU7EncDJyC3v4Mn6IiaE8ncbNlbWKRip-w/edit#slide=id.gd82b0d79b8_1_5). To gain more insight, feel free to go through my [project report](https://docs.google.com/document/d/1kIrIsdWdtzW4BaWf0KIx4pynCHSgu2STvnH3MrIfkbk/edit#).
