# AI Facial Recognition Project

Completed by: Sherief Soliman

The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities.
links for the data sets used

face detection : https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data

emotion detection : https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

Files Included and Use:
1. Data PreProcessing
  data_preprocessing.py: Responsible for preparing the data for training, testing, and validation
  Will generate a preprocessed_data.pth file as output which will prevent the necessity of preprocessing the data each time training occurs. This file can be downloaded from the google drive link provided at the end of this readme. Should this file not be present, simply rerun data_preprocessing.py. 

2. Training:
   train.py: When running this code, this has three arguments as follows
   1: Will train the MainModel
   2: Will train variant 1
   3: Will train variant 2
   e.g the command 'python train.py 1' will train the Main Model
   **Update for part 3**
   There is a commented section in which you can change the dataset you wish to train with. Change the path by uncommenting the path of the dataset to be trained and commenting out the others

   The output of this training will be a .pth file for the model trained. This was done in order to allow for verification of the results reported on.
4. Main Model:
   MainModel.py is the class that will be used for the Main Model. This code is not meant to be run, and is run through train.py
5. Variant 1:
   Variant1.py is the class that will be used for the Variantl. This code is not meant to be run, and is run through train.py
6. Variant 2:
   Variant2.py is the class that will be used for the Variant2. This code is not meant to be run, and is run through train.py
   
7. Model Evaluation: model_evaluation.py can be run to reproduce the evaluation of the models. It expects the presence of the necessary *.pth files saved from Pytorch and the training of the models. Running this will generate popup window with the confusion matrix plots one at a time. The windows will need to be dismissed one at a time, and then the performance metrics will be output in the terminal

8. Model Evaluation 2: model_evaluation_2.py This is focused on the evaluation and use of the main model alone. It will display the performance metrics of the model. In addition, any images can be added to the ./single_image. Running this script will return the prediction of the model on the image(s).
   
9. Gender Data Preprocessing: data_preprocessing_gender.py is used to preprocess the data that is split by gender in the ./images_AI_dataset_gender/... folder. Functions within this script are used in the bias evaluation
10. Bias Evaluation: model_evaluation_gender.py will evaluate the Main Model based on the dataset split by gender. This will return the performance metrics for each gender/class combination. This will also split the dataset if the  female_data_splits.pth, male_data_splits.pth, and non-binary_data_splits.pth.
11. K-Fold Cross Training: kfold_training.py can be run if you wish to train new model using the 10 fold cross validation technique. This will output *.pth files for each fold. It will also output *.npy which store the indices of the test data.
12. K-Fold Evaluation: kfold_evaluation.py will evaluate the performance of the saved folds from the training. If the *.pth and *.npy files are in the root folder, this can be run without training.
13. Synthetic Data Generator: synthetic_data.py was written in case bias was located. This file was not used, but was submitted as a representation of how synthetic images may be generated from the existing images.

An archive folder has the necessary *.pth files should you wish to replace the files in the main directory with new training model. It also contains 10 sample images for each class
   
