# Titanic_Survival_Prediction

## Table of Contents

1. [Introduction](#introduction)
   
2. [Data Preparation and Analysis](#data_analysis)
    a. [EDA - Exploratory Data Analysis](#eda)
    b. [Data Cleaning and Preprocessing](#data_cleaning)
       
3. [Model Selection](#model_selection)
    a. [First step: What is the type of output?](#step1)
    b. [Second step: Abide by the data](#step2)
    c. [Third step: Training time](#step3)


## 1. Introduction <a name="introduction"></a>

This project aims at gaining skills on data cleaning, data analysis, model selection, and model building/tuning.

This project is about the Titanic Survival Rates of a person, given some features. The main objective is to build a well selected model that shall predict which of the Titanic passengers survived, based on their age, gender, passenger class, number of children/siblings/parents onboard, etc.

The project uses two Kaggle datasets, _train_ and _test_. Train is meant to build the model, test and validate it. It includes both the features of a given person (inputs), and a binary describing if that person survived (output). The datasets are sourced from Kaggle, and are stored in the datasets folder.

The **main.ipynb** notebook file contains the source code, as well as some documentation, both theoretical and practical, on the steps used/useful to lead a data science project.
The data analysis plots are available in the data_analysis_plots folder.

## 2. Data Preparation and Analysis <a name="data_analysis"></a>

The aim here is to explore the data:
- What features seem the most important? Which of them seem correlated?
- Are there any missing values? If there are -Spoiler Alert: there are some!- how do we handle them?
- Do we need to prepare the data?

The data must be clean/structured to work on, for the model to predict better. 

### a. EDA - Exploratory Data Analysis <a name="eda"></a>

The EDA step is meant to understand the data. It is meant to identify the useful information provided by the data, understand the data, and evaluate if the are links between the different features.
This can also provide us with some intuition on how the predictions should go - though intuition isn't always right. Sadly, we get the intuition that a 100 years old male passenger in 3rd class won't be classified as survived. Does the data seem to go in that sense? Plots (heatmaps, countplots, etc.) can be useful to determine that. You can find the different plots in the [data_analysis_plots](https://github.com/Gdeterline/Titanic_Survival_Prediction/tree/aaf7590bbfc4bdf2ee6ed5dac0e31c8a516f9290/data_analysis_plots) folder.

### b. Data Cleaning and Preprocessing <a name="data_cleaning"></a>

There are two main compenents here:
1. Data Cleaning: Here, we want to determine the proportion of empty fields/values per feature. And within each feature, we want to know which fields are missing. Fortunately _pandas_' library provides the functions to do so. Once the missing values are identified, we can choose a strategy to tackle the issue - Dropping missing rows, filling them with the mean value for that feature, etc.
This also means labeling the data if it is not. By having a look at the _train_ dataset, this shouldn't be an issue.

2.  Data Preprocessing: Here, we want to prepare the data so that it is easily usable by the model. This can mean doing some feature engineering (create new features from existing features) as well as converting String features to Integers/Floats so that the model can read them.

## 3. Model Selection <a name="model_selection"></a>

Theoretically, the next step would now be to choose the right prediction model for our project. I'll explain the different aspects to bear in mind when choosing a model. But for this project, since the idea is to gain skills in data science, we will go through several different prediction models that apply here.

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



