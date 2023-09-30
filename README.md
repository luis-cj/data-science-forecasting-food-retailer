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

[Feature Selection](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/05_Feature%20selecion.ipynb)

### 6. Modelling

The goal is to check that the model development and evaluation is carried out successfully in order to be automated for production in phase 7 (Model Deployment).

Modelling can be checked in the following Jupyter Notebook:

[Modelling](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/06_Modelling.ipynb)

### 7. Model Deployment

This final phase aims to deliver a simple, easy-to-use production code that calls different functions able to process any new dataset and complete all the tasks done through the entire project (data cleaning, transformation and final use of the selected optimized machine learning algorithm for each product and store).

This part includes:

- **Pre-production code setup**: building the functions for the final deployment/execution code. The goal is to generate a Python Script with all the functions defined during this pre-production code (written in a Jupyter Notebook format).

- **Re-training code**: to keep the model up to date by training it again after a certain period of time. It is ready to be used by the business in case they want to keep improving the model.

- **Deployment code**: it makes use of the functions built during the pre-production phase. This is a Python Script that allows all the models to be executed in a batch process, API or web app. Short and simple script that requires another script with built-in functions that contain all the data pipeline (the pre-production code functions).

Model deployment can be checked in the following Jupyter Notebooks and Python Scripts:

[Pre-production code setup notebook](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/notebooks/07_Production%20code%20setup.ipynb)

[Functions to be used during model deployment](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/python_scripts/functions_retail.py)

[Model re-training code](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/python_scripts/retraining_code.py)

[Model deployment code (Python script to be run in a batch process, API or web app)](https://github.com/luis-cj/data-science-forecasting-food-retailer/blob/main/python_scripts/execution_code.py)

## BONUS: Lessons learnt

- **Intermittent demand can be modelled**: in retail forecasting it is usual to find periods of time of no sales. And that might mean that there's a stock outage or simply that a product is only sold during certain seasons. But that can reduce the performance of an forecasting model. In order to take into account for all that, new variables can be included in the model such as stock outage, lags, or moving windows. Also, adding information about the inventory can be helpful as well.

- **Hierarchical forecasting needs to be solved (reconciliation problem)**: when solving forecasting problems in retail the data is usually divided into multiple levels of aggregation. In this case, it is aggregated by category, sub-category and product. When making predictions, the final result will depend on the aggregation level we are modelling, since the results are not going to match. It is necessary to make predictions at one aggregation level, and then make sure the predictions make sense at all the aggregation levels. In this case, a bottom-up approach could be applied, where we have predictions at product level and then we could just simply aggregate the predictions at category level. It's very simple. However, it is less reliable than modelling at category level (top-down approach) (in this way there is more data for the model, so more patterns can be learnt and eventually end up with better predictions). 

- **Big data modelling needs to be efficient**: there can be tens, hundreds and thousands of categories or products to be modelled in retail forecasting. Then, it is necessary to use fast ML models (in this case LightGBM worked very well).

- **Multi-step forecasting is necessary to predict into the future (more than 1 day ahead)**: from a business perspective it is not very useful to be able to predict only 1 day ahead. In this case we forecasted for a week ahead. And traditional forecasting methods can handle that automatically, but it is not the case of ML models since they need all the records available to make the predictions. Here appears the multi-step forecasting methodology. There are 2 ways to do it. First, different models can be developed to solve for different time frames in the future (which would be called direct modelling). The second way is to do recursive forecasting, where a single ML model is used to predict the same amount of time in the future (only 1 day ahead). With recursive forecasting the model predicts 1 day ahead of all the available data, and then uses these new data to make the forecasting for the following day, and so on. This way is easier to maintain in real projects.