# Data science project: Forecasting Food Retailer
Data science project to forecast retail sales from a large grocery chain using a multi-step forecasting Machine Learning algorithm.

This is a **Machine Learning (ML) project**, where the main goal is to develope a machine learning model able to run in batch mode, API or web app for further model explotation in a real business scenario. It includes data cleaning process, exploratory data analysis, feature engineering, ML modeling and retraining and deployment code.

All the project is carried out using Python Jupyter Notebooks and Python scripts.

## Context
A large grocery chain wants to forecast the sales of 10 food products, sold in 2 of their stores. In order for them to provide the most value, they would like to have a forecast of their sales 8 days in advance, so they can avoid shortage of products and reduce operative costs in their stores.

The business managers have outsourced this project to a data scientist to deliver a machine learning product able to be run in batch mode. In this way the model can be reused at any time in the future and thus keep adding value to the business.

<p align="center">
  <img width="480" height="270" src="https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/images/grocery_store_gif.gif">
</p>


## Methodology

It is required to deliver a final product to the business owners that is capable of predicting sales for each product on each store for 8 days in advanced. In order to do that, the most efficient way is to establish a methodology that allows us to carefully collect the data, clean the data and create an ML model (among other steps) before releasing the final product.

This methodology and all the steps involved in it are run in Python Jupyter Notebooks. Then, the final product is the summary of all the necessary steps in a single Python Script ready to be thrown into production mode for further explotation of the ML model.

These steps are defined under this section with the link to their respective Python Jupyter Notebooks.

<!-- I would like to add at some point if I have time some definition of forecasting and specifically multi-step recursive forecasting. Mention big data modeling capabilities. -->

### 1. Objectives

- Develope a multi-step forecasting ML model that can predict sales for each product on each store within the next following 8 days from the data collection. 
- Create a Python script so the ML model can be automatically run in batch mode, API or web app for further explotation and real use in the company.

### 2. Data collection and setup

As a starting point, a SQL database is provided containing 3 years of selling records for each product and store. Data is provided by the business owners.

Three different datasets are provided, corresponding to:

- Calendar

- Sales

- Prices

All of the datasets are joined as it is shown in the following Jupyter notebook:

[Data setup](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/01_DataCollection_Setup.ipynb)

### 3. Data cleaning

This part involves checking data types and general data quality. The main steps in this phase are cleaning and processing data from missing values and outliers that the original dataset may contain.

Data cleaning can be checked in the following Jupyter Notebook:

[Data cleaning](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/02_DataCleaning.ipynb)

### 3. Exploratory Data Analysis (EDA)

A statistical and graphical analysis of the data allows further understanding of the data and the business problem.

EDA can be checked in the following Jupyter Notebook:

[EDA](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/03_EDA.ipynb)

### 4. Feature Engineering

<!-- All variables that are going to be used in the ML model need to be prepared for that task. Here, any new variables relevant for the problem might be created. Also, a first feature selection is carried out. Finally, different class balancing methods can be evaluated to check whether they're necessary or not. -->

New predictive variables are created in this section in order to make the ML model be more accurate.

Feature Engineering can be checked in the following Jupyter Notebook:

[Feature Engineering](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/04_Feature%Engineering.ipynb)

### 5. Feature Selection


In this project this Feature Selection section does not consist of selecting the final variables that will be included during the modelling part. It is used to check what is the most suitable feature selection method and to have an idea of the number of variables that can get selected for a model.

Feature Selection can be checked in the following Jupyter Notebook:

[Feature Selection](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/05_Feature%selecion.ipynb)

### 6. Modelling

The goal is to check that the model development and evaluation 

Feature Selection can be checked in the following Jupyter Notebook:

[Modelling](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/06_Modelling.ipynb)

### 7. Prepare production scripts

This Notebook aims to make sure that the whole pipeline works correctly before encapsulating everything on the final Python Script. 

Production preparing NOTEBOOK

### 8. Final product 

The final product is here. The production Python Script is short and simple, requiring another script with built-in functions that contain all the data pipeline to execute the forecasting model.

Also, another Python Script for retraining the ML model with more data in the future is ready to be used by the business in case they want to keep improving the model over time.

Production script 

Functions script

Retraining script

## BONUS: Lessons learnt


