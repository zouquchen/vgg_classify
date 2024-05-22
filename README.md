<div align="center">
  
# vgg_classify
  
[English](/README.md) ｜ [简体中文](/README.cn.md) 

</div>

# Custom Dataset
1. Download the dataset from: https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda
2. Split the dataset into training set (train) and testing set (test) in a ratio of 9:1, and store them in the train and test folders respectively.

# Train the Model
1. Open the vgg_train.py file and modify the dataset path to correspond to the paths of the custom dataset's train and test folders.
2. Run the program to start training. No pre-trained model is needed. The model will be saved when the epoch reaches 40 (epochs * milestone[1]).

# Test a Single Image
1. Open the vgg_classify.py file and modify the model path and the path of the image to be detected.
2. Run the program to obtain the detection result.

# Visualization Interface
1. Open the run.py file and modify the model path on line 27.
2. Run the program to open the system interface.
