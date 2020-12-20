# HumanActivityRecognitionLSTM

### About
 
Smart watches and wearables have become increasingly popular, including features that track different activity's, remind a user when it's time to stand, or can tell if you are walking versus running. Many people rely on their smart watches to help them stay healthy and active, highlighting the importance of recognizing human activity from sensor readings.

This projects aims to classify accelerometer and gyroscope readings taken from smartphone sensors into six different activity classes: standing, sitting, laying, walking, walking upstairs, and walking downstairs. This could be useful in tracking how many times a person has gotten up in an hour, or starting a sleep tracking application if the person has been lying down for a long period of time.

### Dependencies

This project uses Google Colaboratory to run the code and Google Drive to store the data. After downloading the data, update the TRAIN_DIR and TEST_DIR variables to point to where the data is stored in Google Drive.

1. Colab getting started: https://colab.research.google.com/notebooks/intro.ipynb
2. Sign up with Google Drive: https://www.google.com/intl/en_in/drive/

The project runs on Python 3 also requires the following libraries: tensorflow, keras, sklearn, numpy, pandas, matplotlib, and seaborn. Dependencies do not need to be installed if running the notebook on Colab.

### Background

This section will provide some background on Recurrent Neural Networks (RNNs), because they are good at understanding time series data and this project uses RNNs to classify the sensor readings. RNNs are different from other types of neural networks in that they account for the dependencies between inputs and are particularly useful in working with sequential data. Most neural networks assume inputs are independent of each other, which doesn't work for sequential data such as a sentence, where previous words can affect future words in the sentence.

The "recurrent" portion of an RNN refers to the process of applying the same operations on multiple inputs over time, allowing the RNN to retain historical information in the sequence and account for time dependencies.

![RNN visualization](/figures/RNN_diagram.png)

Referring to the diagram above, a new input, x, is given to the model at each timestep, t. The model maintains a hidden state, s, which it uses to update the weights, W, U, and V, and output a new prediction, o, at each timestep. This diagram shows the RNN "unfolded" so the reader can view the operations at each timestep. The RNN uses the same weights across different inputs and timesteps to track historical dependencies in the sequence. A non-linear activation is applied on the output and optionally a bias term, B, is included. Refer to the equations below for how the hidden state and output are calculated at each timestep.

S = max(XtU + SW + Bh, 0)
Ot = max(SV + By, 0)

In a standard neural network, back propagation is used to update weights in order to minimize the network's loss. This is done by minimizing the network's cost function and determining how much to tweak the network's weights. Standard networks optimize this process by using stochastic gradient descent, where the "amount" to tweak the weights is known as the gradients. 

RNNs use a modified version of back propagation known as backpropagation through time (BPTT) where the error for each time step is calculated and accumulated backwards in time to update the weights. In this way, the reader can view each timestep as a different "layer" in the RNN.

![BPTT visualization](/figures/BPTT_diagram.png)

RNNs retain state across time to track sequential dependencies and are used commonly to analyze time series data. RNNs can also be used in supervised learning tasks to classify sequences of inputs. For example, let's say we want to classify tomorrow's weather as either rainy or not rainy depending on the last 10 days of precipitation, humidity, and UV index where we know how the weather was over the last 10 days. The input at each time step would be 3 variables (precipitation, humidity, and UV index), and the output would be a predicted class (rainy or not rainy). If we have thousands of sequences of 10-day data, we can feed each sequence individually into the network and use BPTT to update our weights in order to minimize the classification error. This way, we develop a network which can accurately predict if it is going to rain tomorrow or not.

### Related Work

We used the University of California Irvine's (UCI) Human Activity Recognition (HAR) dataset, which was collected in 2012 as part of a research project to better understand human activities from smartphone sensor signals [9]. Since then, a variety of research on this dataset has been conducted to improve classification performance of activities from acceleration and velocity signals.

The original paper published from this dataset in 2013 by Chetty et. al manually curated a list of 561 features from the sensor readings and compared the performance of several supervised classifiers such as SVMs, Naive Bayes, and Random Forests, on the features. They also experimented with an IBk lazy-learner. IBk does not build a model and can avoid expensive training times, but makes predictions by comparing points to their nearest neighbors, which can be slow. Chetty et al evaluated model performance using a 5-fold cross validation on the 10,000 samples collected, inputting the 561-length feature vector and class label to the model to get a prediction. The team found that in terms of efficiency and accuracy, the random forest model performed better than a single decision tree, Naive Bayes, or K-Means, achieving over 96% accuracy with the full feature set. The team also found that an IBk lazy-learner had the best prediction accuracy. Even though the "training" time for this algorithm is low, the prediction time tends to be high, so the random forest model was preferred.

The results in Chetty et al's original work were promising but require a lot of manual feature engineering on the signal data. Since 2013 and the rise of wearable devices, many other researchers have tried to improve on this through the use of neural networks. Ullah et al developed a stacked LSTM approach in 2019. Their network consisted of a ReLu layer to normalize the sensor data, followed by five LSTM layers, and a final softmax layer to output predictions. Ullah et al also used a standard cross entropy loss function with an extra L2 regularization term to add bias into the model and prevent overfitting and an adam optimizer to update weights. They used the UCI HAR dataset, splitting the data into a 70:30 train split where each sample was a 128x9 vector containing 128 time steps and 9 measurements taken per time step. Ullah et al compared their results, achieving over 93% accuracy, to other Markov and SVM models; however they were not able to reproduce the original 96% accuracy presented by Chetty et al in 2013, although they were able to produce results with significantly less feature engineering. When comparing individual class performance of their model, Ullah et al found that the model performed best at predicting when a participant was walking and struggled to differentiate between sitting and standing, but had an overall high precision and recall across all activities.

In addition to stacked LSTMs, other researchers have also proposed using bidirectional LSTMs to further improve results. Bidirectional LSTMs analyze the input sequence going forward and backwards in time, leveraging future inputs to better inform a decision at the current timestep. This approach was proposed by both Hernandez et al in 2019 at the STSIVA Symposium and by Yu et al at the 2018 International Conference on Mechanical, Control and Computer Engineering.

Similar to Ullah et al, Hernandez et al also used an Adam Optimizer to update weights and a grid search approach to search for the optimal number of LSTM layers and neurons per LSTM. They used a learning rate of 0.001 and inputted a feature vector of 128x9 (timesteps x features) to the model. They found that a deeper network with 3 layers and 175 neurons per LSTM performed best, at 92.67% average accuracy after 500 epochs of training.

Similar to Ullah et al, Hernandez et al also found that the model had the most difficulty classifying standing from sitting. Although Hernandez et al's average accuracy of 92.67% was close to Ullah's 93%, it seems that bidirectional layers performed slightly worse than regular, stacked LSTMs.

### Data

The data is available at UCI's machine learning repository: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
In the experiment 30 participants aged 19-48 were asked to perform six activitie while wearing a Samsung Galaxy SII on their waist. While performing the activities sensor data from the accelerometers and gyroscopes from the smartphone were collected. The sensor data was collected at 2.56 second intervals with a 50% overlap, for a total 128 readings per activity.  The published dataset is already pre-processed by applying noise filters to the accelerometer and gyroscope sensor data. Participants were asked to perform the activities multiple times so that researchers could collect over 10,000 samples evenly distributed over each activity. Below you can see the signal readings over time for a sample participant who was asked to stand.

![Sample Reading Visualization](/figures/sample_reading_standing_participant.png)

Walking upstairs and walking downstairs had slightly less samples but overall enough samples were collected from each class to avoid a class imbalance.

![Activity Distribution](/figures/activity_distribution.png)

Each sample consists of three measurements: total acceleration, body acceleration, and gyroscope velocity, where body acceleration was estimated by subtracting gravitational acceleration from the total acceleration. Additionally, each measurement was taken in respect to three dimensions: X, Y, and Z. The combination of measurement and dimension results in 9 features collected for each time step, where 128 time steps were recorded for each activity. A 70/30 split was used for testing and training and measurements for each dimension and sensor were combined to form the following train and test dimensions

1. Train: 7,352 x 128 x 9
2. Test: 2,947 x 128 x 9

### Methods

In this project, we will be exploring multiple variations of recurrent neural networks to train and test the data that was mentioned. The models that we have chosen for this project are: basic LSTM, stacked LSTM, and bidirectional LSTM. Our models use a many to one prediction on the dataset. There are 9 inputs (x, y, z coordinates of the total acceleration, body acceleration, and gyro velocity), and the output is the classification of one of the six activities (walking, walking upstairs, walking downstairs, sitting, standing, laying down).

#### LSTMs

Basic RNNs suffer from short-term memory, which means that when a sequence becomes too long, it will be harder for the model to carry information from earlier timesteps. Basic RNNs will leave out information in the beginning, which is not ideal. The solution to this problem is the introduction of LSTMs. LSTMs have an internal hidden mechanism called gates that regulate the flow of information inputted into the model. These gates are able to learn which data in a sequence is important enough to keep and what to leave out. The cell state and its various gates (forget gate, input gate, output gate) are core concepts for the LSTM. Relevant information is transported through the cell state throughout the processing of the sequence. As information gets passed through, the neural network decides which information to keep and which to leave out. The risk of having vanishing gradients is also decreased. Basic LSTMs have a single hidden layer of LSTM units. They can support multiple parallel sequences of input data, such as the accelerometer and gyroscope data in our dataset.

1. Input Gates: Updates the cell state
2. Forget Gates: Decides what information to be saved and which to be left out
3. Output Gates: Decides what the next hidden state should be

#### Stacked LSTMs

In the history of neural networks, many types of networks have seen improved performance by increasing depth, leading to the rise of deep neural networks. Depth is added to the network by adding intermediate nonlinear hidden layers between inputs and outputs, for example by adding many alternating convolution and pooling layers in a CNN. Increasing depth in the network allows deeper layers to further abstract information and learn higher-level features. For example, early layers in a CNN may extract edges while deeper layers extract fine-grained information specific to the image. This same concept has been explored for LSTMs as well to see if increasing depth leads to better performance.

Pescanu et al explored different types of depth in an RNN and originally coined the term “Stacked LSTM” in 2014. They explored how depth exists in an RNN because each timestep can be seen as a layer in the network, but these connections are shallow in that “there exists no intermediate, nonlinear hidden layers” between timesteps. In order to introduce true depth into the network, they explored stacking multiple recurrent hidden layers on top of each other. In this model, LSTM layers are ordered sequentially, where each previous layer passes a sequence of values, one output per input timestep, to the next layer. This introduces intermediate nonlinear hidden layers between each recurrent unit so the network can focus on the data at different timescales.

This method was explored by Graves, Mohamed, and Hinton in 2013 when they applied it to speech recognition. Graves et al experimented with 1-5 stacked LSTM layers, 250-622 hidden units per LSTM, training a little over a 100 epochs with an Adam Optimizer on recorded, sequential voice input. They discovered that performance improved by adding more layers but degraded after the network was deeper than 5 layers and that network depth had more effect on performance than layer size (number of hidden units used).

#### Bidirectional LSTMs

Bidirectional LSTMs analyze an input sequence from both directions - going forward and backwards in time, to make better predictions at the current time step. This may seem counterintuitive at first but imagine a movie review. Being able to see what the next 10 words in the review may help you better determine if the person is content or discontent with the movie. Expanding this example to our time series activity data, imagine we are just analyzing acceleration in the x-dimension. Looking at the last 10 readings, we may see acceleration is increasing and can use this information to guess the participant is walking forward. But the next 10 readings show decreasing acceleration, so it turns out that the participant was actually about to sit down. If our network is able to look ahead a few readings into the future, it can make better predictions about what activity the participant is currently performing. This bidirectional dependency exists in our time series data because future sensor readings can inform present ones in the same way that past readings do.

Bidirectional properties on human activity data was explored by both Hernandez et al and Yu et al where they were able to achieve almost 93% average accuracy in classifying the six types of activities in the dataset. Bidirectional LSTMs are often combined with deep, stacked LSTMs to further improve performance.

### Results

To improve performance on a base LSTM various hyperparameters such as number of hidden units in the LSTM, epochs, and dropout level were tuned. LSTMs use a hidden state to track sequential dependencies in the input, our model uses dropout to prevent overfitting, and number of epochs is used to determine how long to train the model for. Models were trained on Google Colab with GPU support where the average epoch took about 1 second, so testing over a wide range of epochs was inexpensive, taking at most 10 minutes. We tested average accuracy over various hyperparameter combinations to find that 150 hidden units, trained over 512 epochs with a 75% dropout level produced the best results.

![LSTM Accuracy Over Unit Epoch](/figures/LSTM_accuracy_over_unit_epoch.png)

![LSTM Accuracy Over Dropout](/figures/LSTM_accuracy_over_dropout.png)

![LSTM Accuracy Over Epoch](/figures/LSTM_accuracy_over_epoch.png)

### Citations

1. Chetty, Girija, Matthew White, and Farnaz Akther. "Smartphone based data mining for human activity recognition." Procedia Computer Science 46 (2015): 1181-1187.
M. Ullah, H. Ullah, S. D. Khan and F. A. Cheikh, "Stacked Lstm Network for Human Activity Recognition Using Smartphone Data," 2019 8th European Workshop on Visual Information Processing (EUVIP), Roma, Italy, 2019, pp. 175-180, doi: 10.1109/EUVIP47703.2019.8946180.
2. F. Hernández, L. F. Suárez, J. Villamizar and M. Altuve, "Human Activity Recognition on Smartphones Using a Bidirectional LSTM Network," 2019 XXII Symposium on Image, Signal Processing and Artificial Vision (STSIVA), Bucaramanga, Colombia, 2019, pp. 1-5, doi: 10.1109/STSIVA.2019.8730249.
3. S. Yu and L. Qin, "Human Activity Recognition with Smartphone Inertial Sensors Using Bidir-LSTM Networks," 2018 3rd International Conference on Mechanical, Control and Computer Engineering (ICMCCE), Huhhot, 2018, pp. 219-224, doi: 10.1109/ICMCCE.2018.00052.
4. A. Graves, A. Mohamed and G. Hinton, "Speech recognition with deep recurrent neural networks," 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, Vancouver, BC, 2013, pp. 6645-6649, doi: 10.1109/ICASSP.2013.6638947.
5. Pascanu, R., Gulcehre, C., Cho, K., & Bengio, Y. (2014). How to construct deep recurrent neural networks. In Proceedings of the Second International Conference on Learning Representations (ICLR 2014)
5. Britz, Denny. “Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs.” WildML, 8 July 2016, www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/.Recurrent Neural Network Diagram
6. Britz, Denny. “Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients.” WildML, 1 Apr. 2016, www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/. BPTT Diagram
7. Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
8. Phi, Michael. “Illustrated Guide to LSTM's and GRU's: A Step by Step Explanation.” Medium, Towards Data Science, 28 June 2020, towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21. 
9. X. Shi, Z. Chen, H. WAng, DY. Yeung, WK. Wong, WC. Woo, “Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting”,Department of Computer Science and Engineering of Hong Kong University of Science and Technology, 2015 https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf
10. Brownlee, J. (2020, August 27). LSTMs for Human Activity Recognition Time Series Classification. Retrieved December 20, 2020, from https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/

