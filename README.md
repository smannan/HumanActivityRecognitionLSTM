# HumanActivityRecognitionLSTM

### About
 
Smart watches and wearables have become increasingly popular, including features that track different activity's, remind a user when it's time to stand, or can tell if you are walking versus running. Many people rely on their smart watches to help them stay healthy and active, highlighting the importance of recognizing human activity from sensor readings.

This projects aims to classify accelerometer and gyroscope readings taken from smartphone sensors into six different activity classes: standing, sitting, laying, walking, walking upstairs, and walking downstairs. This could be useful in tracking how many times a person has gotten up in an hour, or starting a sleep tracking application if the person has been lying down for a long period of time.

### Dependencies

This project uses Google Colaboratory to run the code and Google Drive to store the data. After downloading the data, update the TRAIN_DIR and TEST_DIR variables to point to where the data is stored in Google Drive.

1. Colab getting started: https://colab.research.google.com/notebooks/intro.ipynb
2. Sign up with Google Drive: https://www.google.com/intl/en_in/drive/

The project runs on Python 3 also requires the following libraries: tensorflow, keras, sklearn, numpy, pandas, matplotlib, and seaborn. Dependencies do not need to be installed if running the notebook on Colab.

### Data

The data is available at UCI's machine learning repository: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
In the experiment 30 participants aged 19-48 were asked to perform six activitie while wearing a Samsung Galaxy SII on their waist. While performing the activities sensor data from the accelerometers and gyroscopes from the smartphone were collected. The sensor data was collected at 2.56 second intervals with a 50% overlap, for a total 128 readings per activity.  The published dataset is already pre-processed by applying noise filters to the accelerometer and gyroscope sensor data. Participants were asked to perform the activities multiple times so that researchers could collect over 10,000 samples evenly distributed over each activity. Below you can see the signal readings over time for a sample participant who was asked to stand.

[!Sample Reading Visualization](figures/sample_reading_standing_participant.png)

Walking upstairs and walking downstairs had slightly less samples but overall enough samples were collected from each class to avoid a class imbalance.

[!Activity Distribution](figures/activity_distribution.png)

Each sample consists of three measurements: total acceleration, body acceleration, and gyroscope velocity, where body acceleration was estimated by subtracting gravitational acceleration from the total acceleration. Additionally, each measurement was taken in respect to three dimensions: X, Y, and Z. The combination of measurement and dimension results in 9 features collected for each time step, where 128 time steps were recorded for each activity. A 70/30 split was used for testing and training and measurements for each dimension and sensor were combined to form the following train and test dimensions

1. Train: 7,352 x 128 x 9
2. Test: 2,947 x 128 x 9
