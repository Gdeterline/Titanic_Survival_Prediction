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

5. [Prediction Models](#prediction_models)

   a. [Logistic Regression](#logistic_regression)
   
      1. [Model Training](#lr_model_training)
         
      2. [Evaluation Metrics](#lr_evaluation_metrics)

   b. [K-nearest neighbors](#knn)

      1. [Model Training](#knn_model_training)
         
      2. [Evaluation Metrics](#knn_evaluation_metrics)

   c. [K-means](#kmeans)

      1. [Model Training](#km_model_training)
         
      2. [Evaluation Metrics](#km_evaluation_metrics)

   d. [Decision trees](#decision_trees)

      1. [Model Training](#dt_model_training)
         
      2. [Evaluation Metrics](#dt_evaluation_metrics)

   e. [Random Forest](#random_forest)

      1. [Model Training](#rf_model_training)
         
      2. [Evaluation Metrics](#rf_evaluation_metrics)
  
      3. [Hyperparameters tuning and cross validation](#rf_tuning_cv)
  
   f. [SVM - Support-vector machines](#svm)
         
      1. [Model Training](#svm_model_training)
         
      2. [Evaluation Metrics](#svm_evaluation_metrics)
  
   g. [Neural Networks](#neural_networks)

      1. [Model Training](#nn_model_training)
         
      2. [Evaluation Metrics](#nn_evaluation_metrics)
  
   h. [Naive Bayes](#naive_bayes)

      1. [Model Training](#nb_model_training)
         
      2. [Evaluation Metrics](#nb_evaluation_metrics)
  
   i. [Gradient Boosting](#gradient_boosting)

      1. [Model Training](#gb_model_training)
         
      2. [Evaluation Metrics](#gb_evaluation_metrics)
  
6. [Conclusion & Thougts](#conclusion)



## 1. Introduction <a name="introduction"></a>

**Motivation of the project:** I am really passionate about data and how to get the best out of it. And what is better than seeing what the data means today, and perhaps what it could lead to tomorrow? Data tells a story, that asks nothing else than to be continued! So how about we make prediction models to make the data talk? That is what I want to do.
Therefore, this project aims at gaining skills on data cleaning, data analysis, model selection, and model building/tuning.


This project is about the Titanic Survival Rates of a person, given some features. The main objective is to build a well selected model that shall predict which of the Titanic passengers survived, based on their age, gender, passenger class, number of children/siblings/parents onboard, etc.

The project uses two Kaggle datasets, _train_ and _test_. Train is meant to build the model, test and validate it. It includes both the features of a given person (inputs), and a binary describing if that person survived (output). The datasets are sourced from Kaggle, and are stored in the datasets folder.

The **main.ipynb** notebook file contains the source code, as well as some documentation, both theoretical and practical, on the steps used/useful to lead a data science project.
The data analysis plots are available in the data_analysis_plots folder.

## 2. Ressources <a name="ressources"></a>

This project uses several tools/libraries that are required to doing data science. This was the perfect occasion to learn/master them. Here they are listed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn





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
This also means labeling the data if it is not. By having a look at the _train_ dataset, this shouldn't be an issue.

2.  Data Preprocessing: Here, we want to prepare the data so that it is easily usable by the model. This can mean doing some feature engineering (create new features from existing features) as well as converting String features to Integers/Floats so that the model can read them. You also want to divide the _train_ dataset into two separate datasets: one to train the model, one to validate it on a test subset. It several cases, it can be useful to shuffle the dataset before doing so, in order to properly evaluate the model performance.

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

## 5. Prediction models <a name="prediction_models"></a>

### a. Logistic Regression <a name="logistic_regression"></a>

#### i. Model Training <a name="lr_model_training"></a>

**Logistic Regression Characteristics:**
- **Output Type**: Predicts probabilities between 0 and 1
- **Use Case**: Used for classification problems. Ideal for binary problems
- **Equation Form**: $p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n)} }$
- **Model Output**: Outputs a probability, then converts it to a class (0 or 1)
- **Range of Predictions**: Outputs values between 0 and 1 (probabilities).

Based on the theory, we train the model provided by scikit-learn **sklearn.linear_model.LogisticRegression()** on the training subset of the _train_ dataset. 
The model has been trained twice:
- First with a single feature, to test out the model and see how things work. The results aren't expected to be very good.
- Then on the several relevant features, to have a complete model. Of course, this is the model we choose to keep.

#### ii. Evaluation Metrics <a name="lr_evaluation_metrics"></a>

Once we are done, we are able to test the model on the test subset.
The accuracy scores for the training and testing subsets can be found in the main.ipynb file. The analysis of the performance too -under/overfitting.

There are several interesting metrics to evaluate the model performance:
- **Confusion Matrix:** it is a table with two rows and two columns (since our problem is binary) that reports the number of true positives, false negatives, false positives, and true negatives. This matrix allows us to calculate:
   - **Precision:** The number of True Positives divided by the number of True Positives and False Positives. It is the ability of the classifier not to label as positive a sample that is negative: $Precision = \frac{TP}{TP + FP}$ 
   - **Recall:** The number of True Positives divided by the number of True Positives and the number of False Negatives. It is the ability of the classifier to find all the positive samples: $Recall = \frac{TP}{TP + FN}$
   - **F1 Score:** The weighted average of Precision and Recall. It takes both false positives and false negatives into account. It is the harmonic mean of the precision and recall: $F1\ Score = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$

- **AUC-ROC (Area under the Receiver Operating Characteristic Curve):** The ROC curve is the plot of the true positive rate (TPR) against the false positive rate (FPR) at each threshold setting.

All the figures can be found in the main file, as well as an analysis for them all.
There are of course other useful metrics, but let's keep some fun for what comes next! ^^

### b. K-nearest neighbors <a name="knn"></a>

#### i. Model Training <a name="knn_model_training"></a>

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

Based on the theory, we train the model provided by scikit-learn **sklearn.neighbors.KNeighborsClassifier()** on the training subset of the _train_ dataset. 

#### ii. Evaluation Metrics <a name="knn_evaluation_metrics"></a>



### c. K-means <a name="kmeans"></a>

#### i. Model Training <a name="km_model_training"></a>

#### ii. Evaluation Metrics <a name="km_evaluation_metrics"></a>



### d. Decision trees <a name="decision_trees"></a>

#### i. Model Training <a name="dt_model_training"></a>

#### ii. Evaluation Metrics <a name="dt_evaluation_metrics"></a>



### e. Random Forest <a name="random_forest"></a>

#### i. Model Training <a name="rf_model_training"></a>

#### ii. Evaluation Metrics <a name="rf_evaluation_metrics"></a>



### f. SVM - Support-vector machines <a name="svm"></a>

#### i. Model Training <a name="svm_model_training"></a>

#### ii. Evaluation Metrics <a name="svm_evaluation_metrics"></a>



### g. Neural Networks <a name="neural_networks"></a>

Okay, this may not be the most logical thing to do now -gradient boosting, naive Bayes, should probably be done first- 
but I was working on building a Neural Network from scratch in parallel so I really wanted to do some neural network model prediction ^^

#### i. Model Training <a name="nn_model_training"></a>



#### ii. Evaluation Metrics <a name="nn_evaluation_metrics"></a>



### h. Naive Bayes <a name="naive_bayes"></a>

#### i. Model Training <a name="nb_model_training"></a>

##### Testing out the neural network buit in the [Building a neural network from scratch project](#https://github.com/Gdeterline/Neural-Network-Build)





#####

#### ii. Evaluation Metrics <a name="nb_evaluation_metrics"></a>



### i. Gradient Boosting <a name="gradient_boosting"></a>

#### i. Model Training <a name="gb_model_training"></a>

#### ii. Evaluation Metrics <a name="gb_evaluation_metrics"></a>



### 6. Conclusion & Thougts <a name="conclusion"></a>

