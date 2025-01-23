# Titanic Survival Prediction

**Welcome to my Titanic Survival Prediction personnal data science project!** This README file contains information/documentation about what is done in the main.ipynb file. The latter is documented and contains plots, tables, etc. so it should be easily readable.

## Table of Contents

1. [Introduction](#introduction)

2. [Ressources](#ressources)
   
3. [Data Preparation and Analysis](#data_analysis)
   
    a. [EDA - Exploratory Data Analysis](#eda)
   
    b. [Data Cleaning and Preprocessing](#data_cleaning)
   
4. [Theoretical Model Selection](#model_selection)
   
    a. [First step: What is the type of output?](#step1)
   
    b. [Second step: Abide by the data](#step2)
   
    c. [Third step: Training time](#step3)

5. [Training and Testing](#training_and_testing)

6. [Prediction Models](#prediction_models)

   a. [Logistic Regression](#logistic_regression)

   b. [K-nearest neighbors](#knn)

   c. [Decision trees](#decision_trees)

   d. [Random Forest](#random_forest)
  
   e. [SVM - Support-vector machines](#svm)
  
   f. [Neural Networks](#neural_networks)

   g. [Naive Bayes](#naive_bayes)

   h. [Gradient Boosting](#gradient_boosting)

6. [Conclusion & Thougts](#conclusion)



## 1. Introduction <a name="introduction"></a>

**Motivation of the project:** I am really passionate about data and how to get the best out of it. And what is better than seeing what the data means today, and perhaps what it could lead to tomorrow? Data tells a story, that asks nothing else than to be continued! So how about we make prediction models to make the data talk? That is what I want to do.
Therefore, this project aims at gaining skills on data cleaning, data analysis, model selection, and model building/tuning.


This project is about the Titanic Survival Rates of a person, given some features. The main objective is to build a well selected model that shall predict which of the Titanic passengers survived, based on their age, gender, passenger class, number of children/siblings/parents onboard, etc.

The project uses two Kaggle datasets, _data_ and _test_. Data is meant to build the model, test and validate it. It includes both the features of a given person (inputs), and a binary describing if that person survived (output). The datasets are sourced from Kaggle, and are stored in the datasets folder.

The **main.ipynb** notebook file contains the source code, as well as some documentation, both theoretical and practical, on the steps used/useful to lead a data science project.
The data analysis plots are available in the data_analysis_plots folder.

## 2. Ressources <a name="ressources"></a>

This project uses several tools/libraries that are required to doing data science. This was the perfect occasion to learn/master them. Here they are listed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- PyTorch



## 3. Data Preparation and Analysis <a name="data_analysis"></a>

The aim here is to explore the data:
- What features seem the most important? Which of them seem correlated?
- Are there any missing values? If there are -Spoiler Alert: there are some!- how do we handle them?
- Do we need to prepare the data?

The data must be clean/structured to work on, for the model to predict better. 

#### a. EDA - Exploratory Data Analysis <a name="eda"></a>

The EDA step is meant to understand the data. It is meant to identify the useful information provided by the data, understand the data, and evaluate if the are links between the different features.
This can also provide us with some intuition on how the predictions should go - though intuition isn't always right. Sadly, we get the intuition that a 100 years old male passenger in 3rd class won't be classified as survived. Does the data seem to go in that sense? Plots (heatmaps, countplots, etc.) can be useful to determine that. You can find the different plots in the [data_analysis_plots](https://github.com/Gdeterline/Titanic_Survival_Prediction/tree/aaf7590bbfc4bdf2ee6ed5dac0e31c8a516f9290/data_analysis_plots) folder.

#### b. Data Cleaning and Preprocessing <a name="data_cleaning"></a>

There are two main compenents here:
1. Data Cleaning: Here, we want to determine the proportion of empty fields/values per feature. And within each feature, we want to know which fields are missing. Fortunately _pandas_' library provides the functions to do so. Once the missing values are identified, we can choose a strategy to tackle the issue - Dropping missing rows, filling them with the mean value for that feature, etc.
This also means labeling the data if it is not. By having a look at the _data_ dataset, this shouldn't be an issue.

2.  Data Preprocessing: Here, we want to prepare the data so that it is easily usable by the model. This can mean doing some feature engineering (create new features from existing features) as well as converting String features to Integers/Floats so that the model can read them. Then, it is important to scale the data so that the model can work on it - some models are sensitive to the scale of the data. You also want to divide the _data_ dataset into two separate datasets: one to train the model, one to validate it on a test subset. It several cases, it can be useful to shuffle the dataset before doing so, in order to properly evaluate the model performance.

## 4. Model Selection <a name="model_selection"></a>

Theoretically, the next step would now be to choose the right prediction model for our project. Let's explain the different aspects to bear in mind when choosing a model. 

#### a. **First step:** What is the type of output? <a name="step1"></a>

1. Is the data **labeled**? In that case, we may apply **supervised learning** algorithms.
Do we want to predict continuous numeric values? Do we want to determine distinct categories? We should then choose between regression, clustering, etc.

2. Is the data **unlabeled**? In that case, we may apply **unsupervised learning** algorithms.
Do we want to group similar data points (Clustering)? And so on.

Do I want to simplify the data? In which case we'll probably do some dimensionality reduction before applying an algorithm, etc.

#### b. **Second step:** Abide by the data <a name="step2"></a>

We have to consider the data cleanliness before choosing the model. 
- does my data lack structure, etc.? - unsupervised learning may provide better predictions
- is my data well structured? - supervised learning should provide good predicitons

Choose a model adapted to the linearity of your project. Linear models won't be very effective on non-linear data.

#### c. **Third step:** Training time <a name="step3"></a>

Consider the time you can allocate to the training of the model: if you need fast results, you'll probably choose a quicker model -regression among others- but with lower quality results. If you have time, you can choose to spend time on the data preprocessing, training, etc. and expect high accuracy for your model predictions.

**For this project, since the idea is to gain skills in data science, and therefore increase my knwoledge on different prediction models, we will go through several different prediction models that apply here.**

<ins>Nota bene:</ins> For what is following, in every case we should do some hyperparameter tuning and some cross-validation. We will do these in only a few cases. Indeed for Logistic Regression for instance, hyperparameter tuning isn't as important as it can be for Random Forest or Decision trees.

## 4. Training and Testing <a name="training_and_testing"></a>

The data is divided into two subsets: the training subset and the testing subset. The training subset is used to train the model, while the testing subset is used to evaluate the model performance - how well it generalizes to new data. The testing subset is meant to be unseen by the model, so that we can evaluate its performance on new data.

## 5. Evaluation Metrics <a name="lr_evaluation_metrics"></a>

Once the training is done, we are able to test the model on the test subset.

There are several interesting metrics to evaluate the model performance:
- **Accuracy:** The number of correct predictions made by the model divided by the total number of predictions. It is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations: $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
- **Confusion Matrix:** it is a table with two rows and two columns (since our problem is binary) that reports the number of true positives, false negatives, false positives, and true negatives. This matrix allows us to calculate:
- **Precision:** The number of True Positives divided by the number of True Positives and False Positives. It is the ability of the classifier not to label as positive a sample that is negative: $Precision = \frac{TP}{TP + FP}$ 
- **Recall:** The number of True Positives divided by the number of True Positives and the number of False Negatives. It is the ability of the classifier to find all the positive samples: $Recall = \frac{TP}{TP + FN}$
- **F1 Score:** The weighted average of Precision and Recall. It takes both false positives and false negatives into account. It is the harmonic mean of the precision and recall: $F1\ Score = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$
- **AUC-ROC (Area under the Receiver Operating Characteristic Curve):** The ROC curve is the plot of the true positive rate (TPR) against the false positive rate (FPR) at each threshold setting.

All the figures can be found in the main file, as well as an analysis for them all.


## 6. Prediction models <a name="prediction_models"></a>

### a. Logistic Regression <a name="logistic_regression"></a>

**Logistic Regression Characteristics:**
- **Output Type**: Predicts probabilities between 0 and 1
- **Use Case**: Used for classification problems. Ideal for binary problems
- **Equation Form**: $p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n)} }$
- **Model Output**: Outputs a probability, then converts it to a class (0 or 1)
- **Range of Predictions**: Outputs values between 0 and 1 (probabilities).


### b. K-nearest neighbors <a name="knn"></a>

 **K-Nearest Neighbors (KNN) Characteristics:**
- **Output Type**: Predicts the class of a data point based on the classes of its nearest neighbors.
- **Use Case**: Used for classification in our case.
- **Parameters**: 
   - **k**: The number of nearest neighbors to consider.
   - **Distance Metric**: Euclidean distance is sufficient in our case. Could be another distance.
- **Model Output**: For classification, assigns the most common class among the k nearest neighbors.
- **Model Characteristics**:
   - **Non-parametric**: Does not assume any specific form for the data distribution.
   - **Instance-based**: Stores all training data and makes predictions based on them directly, which can be computationally expensive.
- **Sensitivity to Data**: Performance depends heavily on:
   - **Value of k**: A small k can lead to noise sensitivity, while a large k may result in oversmoothing.


### c. Decision trees <a name="decision_trees"></a>

**Decision Trees Characteristics:**
- **Output Type**: Outputs a class label for each data point.
- **Use Case**: Used for classification in our case.
- **Model Output**: Outputs a class label for each data point.
- **Model Type**: A tree structure where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents the outcome.
- **Model Characteristics**:
   - **Non-parametric**: Does not assume any specific form for the data distribution.
   - **Interpretable**: Easy to understand and visualize.
- **Model Training**:
   - **Splitting Criteria**: The decision tree algorithm makes splits at each node based on a criterion (e.g., Gini impurity, entropy) that maximizes the information gain.
   - **Stopping Criteria**: The algorithm stops splitting the tree based on a stopping criterion (e.g., maximum depth, minimum samples per leaf).
- **Model Complexity**:
   - **Overfitting**: Decision trees can easily overfit the training data, so it is important to tune the hyperparameters to avoid this issue.
   - **Underfitting**: Decision trees can also underfit the data if they are too shallow or have too few nodes.



### d. Random Forest <a name="random_forest"></a>

**Random Forest Characteristics:**
- **Output Type**: Outputs a class label for each data point.
- **Use Case**: Used for classification in our case.
- **Model Output**: Outputs a class label for each data point.
- **Model Type**: An ensemble model that consists of multiple decision trees.
- **Model Characteristics**:
   - **Non-parametric**: Does not assume any specific form for the data distribution.
   - **Interpretable**: Easy to understand and visualize.
   - **Ensemble Model**: Combines the predictions of multiple decision trees to improve the model's performance.
- **Model Training**:
   - **Bootstrapping**: Random Forest usually uses bootstrapping to create multiple training datasets from the original dataset.
   - **Feature Randomness**: Random Forest uses feature randomness to select a subset of features at each split.
   - **Voting**: Random Forest uses majority voting to make the final prediction.
- **Model Complexity**:
   - **Overfitting**: Random Forest can overfit the training data if the trees are too deep or if there are too many trees in the forest.
   - **Underfitting**: Random Forest can underfit the data if the trees are too shallow or if there are too few trees in the forest.


### e. SVM - Support-vector machines <a name="svm"></a>

**Support Vector Machines (SVM) Characteristics:**
- **Output Type**: Outputs a class label for each data point.
- **Use Case**: Used for classification in our case.
- **Model Output**: Outputs a class label for each data point.
- **Model Type**: A linear model that finds the hyperplane that best separates the classes.
- **Model Characteristics**:
   - **Parametric**: Assumes a specific form for the data distribution.
   - **Margin Maximization**: SVM finds the hyperplane that maximizes the margin between the classes.
- **Model Training**:
   - **Kernel Trick**: SVM can use the kernel trick to transform the data into a higher-dimensional space to make it linearly separable.
   - **Regularization**: SVM uses regularization to prevent overfitting - this consists in adding a penalty term to the loss function.
- **Model Complexity**:
   - **Limits of the Model**: SVM can struggle with large datasets and high-dimensional data. Also, if the data presents unbalanced classes, SVM may not be the best model to use.


### f. Neural Networks <a name="neural_networks"></a>

Okay, this may not be the most logical thing to do now -gradient boosting, naive Bayes, should probably be done first- 
but I was working on building a Neural Network from scratch in parallel so I really wanted to do some neural network model prediction ^^

**Neural Networks Characteristics:**
- **Output Type**: Outputs a class label for each data point.
- **Use Case**: Used for classification in our case.
- **Model Output**: Outputs a class label for each data point.
- **Model Type**: A network of interconnected nodes that can learn complex patterns in the data.
- **Model Characteristics**:
   - **Non-linear**: Can learn non-linear patterns in the data.
   - **Deep Learning**: Neural networks with multiple hidden layers are known as deep learning models.
- **Model Training**:
   - **Backpropagation**: Neural networks use backpropagation to update the weights and biases during training.
   - **Activation Functions**: Neural networks use activation functions to introduce non-linearity into the model.
- **Model Complexity**:
   - **Overfitting**: Neural networks can easily overfit the training data if they are too complex or if they are trained for too many epochs.
   - **Underfitting**: Neural networks can underfit the data if they are too simple or if they are trained for too few epochs.
- **Model Hyperparameters**:
   - **Number of Layers**: The number of hidden layers and the number of nodes in each layer.
   - **Activation Functions**: The activation functions used in the hidden layers.
   - **Learning Rate**: The learning rate used during training.
- **Neural Network Types**:
   - **Feedforward Neural Networks**: The simplest type of neural network where the connections between nodes do not form a cycle.
   - **Convolutional Neural Networks (CNNs)**: Neural networks that are designed to work with image data.
   - **Recurrent Neural Networks (RNNs)**: Neural networks that are designed to work with sequential data.

In our case, we will use a deep and dense neural network with several hidden layers (2/3 layers maximum) and a ReLU activation function for the hidden layers and a sigmoid activation function for the output layer. Indeed, it is not necessary to choose a very complex model for this problem, as the data is not very complex, and the number of features is not very high (in which case a convolutional neural network would be more appropriate).

To do so, we will use a model from PyTorch, a deep learning library that provides a lot of tools to build and train neural networks. We will use the `torch.nn` module to define the neural network architecture and the `torch.optim` module to define the optimizer.

##### Testing out the neural network built in the [Building a neural network from scratch project](#https://github.com/Gdeterline/Neural-Network-Build)

Here, we will use the neural network model that we built from scratch in another project. 
As we have already tested it on the Breast Cancer Wisconsin dataset, this should be interesting to see how well it performs on another dataset.

### g. Naive Bayes <a name="naive_bayes"></a>


### h. Gradient Boosting <a name="gradient_boosting"></a>


### 6. Conclusion & Thougts <a name="conclusion"></a>

