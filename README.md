# Twitter-Bot-Identification
Project Overview
This project involves the analysis of Twitter user data to distinguish between human and non-human (bot) accounts. The goal is to identify patterns in behavior and metadata that allow for the classification of these profiles. The project follows a Big Data Analytics Lifecycle approach, leveraging multiple machine learning models and visualization techniques to achieve this goal.

Contents
Introduction
Task 1: Big Data Analytics Project Design
Task 2: Data Processing and Algorithm Application
Task 3: Data Visualization and Evaluation
Task 4: Multimodal Analysis and Recommendations
Introduction
The aim of this project is to apply Big Data Analytics techniques to the Twitter user dataset. By using machine learning models, we aim to distinguish between human and bot profiles based on various features including tweet frequency, metadata, and text sentiment.

Task 1: Big Data Analytics Project Design
1.1 PHASE 1: DISCOVERY

Objective: Identify Twitter profiles as Human or Bot based on user activity and behavior patterns.
Stakeholders: Social media engineers, researchers.
Tools Used: Python, scikit-learn, pandas, NLTK.
1.2 PHASE 2: DATA PREPARATION

Data Cleaning and Handling Missing Values: Used median imputation for numerical columns and most frequent value for categorical fields.
Feature Engineering: Created account age, bot likelihood scores, and normalized numerical features.
Text Vectorization: TF-IDF applied to text data for machine learning models.
1.3 PHASE 3: MODEL PLANNING

Target Variable: Classify accounts as Human (e.g., male/female) or Bot (e.g., brand).
Algorithms Considered: Logistic Regression, Random Forest, SVM, XGBoost.
Model Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
1.4 PHASE 4: MODEL BUILDING

Data Splitting: 80-20 train-test split.
Model Training: Logistic Regression, Random Forest, and SVM were trained.
Hyperparameter Tuning: Applied RandomizedSearchCV for Random Forest.
1.5 PHASE 5: COMMUNICATING RESULTS

Visualizations: Feature importance, confusion matrix, ROC curve.
Tables and Graphs: Accuracy, precision, recall, and F1-Score.
1.6 PHASE 6: OPERATIONALIZATION

Deployment Plan: Real-time classification of Twitter profiles.
Monitoring: Regularly retrain the model with new data.
Task 2: Data Processing and Algorithm Application
Data Preparation

Cleaned missing data using median and most frequent values.
Engineered features such as account age and bot likelihood scores.
Split the data into training (80%) and testing (20%).
Models Trained

Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
Hyperparameter Tuning

Applied RandomizedSearchCV to optimize Random Forest performance, achieving 79.1% accuracy.
XGBoost Model

Trained an additional XGBoost model, achieving 74.76% accuracy. XGBoost handled class imbalance effectively using the scale_pos_weight parameter.
Task 3: Data Visualization and Evaluation
Top 10 Important Features (Random Forest)
Class Distribution (Human vs Bot)
Correlation Heatmap for Numerical Features
Distribution of Favorite Count and Tweet Count (with Limited X-axis)
Pair Plot for Key Features
Word Cloud for Human and Bot Tweets
Confusion Matrix (XGBoost)
PCA and t-SNE Plots for Dimensionality Reduction
Task 4: Multimodal Analysis and Recommendations
Text Processing and Tokenization

Tokenized and cleaned tweets for sentiment analysis and TF-IDF vectorization.
Sentiment Analysis

Used NLTK's SentimentIntensityAnalyzer to assess the emotional tone of tweets.
TF-IDF Vectorization

Transformed text data into numerical features for machine learning.
Sidebar Color Analysis

Analyzed common color trends in bot profiles to identify branding patterns.
Suggestions for Amendments

Developed recommendations for amending bot-like profiles based on text sentiment and visual patterns.
Conclusion
The tuned Random Forest model delivered the best performance in classifying human and bot profiles, with XGBoost also providing solid results. The data-driven insights, coupled with visualization and model outputs, provide a clear path for identifying and amending bot-like profiles on Twitter.


