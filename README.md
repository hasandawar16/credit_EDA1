# credit_EDA1
📊 Credit Risk Analysis using Exploratory Data Analysis (EDA)
📌 Project Overview

This project performs Exploratory Data Analysis (EDA) on loan application data to understand customer behavior and identify factors influencing loan default risk. The analysis involves data cleaning, handling missing values, feature engineering, visualization, and merging historical loan records to extract meaningful insights for credit risk assessment.
This project focuses on data understanding and insights, not prediction or machine learning.

📂 Datasets Used

application_data.csv
Contains current loan application details of customer
Includes demographic, financial, and loan-related attributes
Target variable indicates loan repayment status
previous_application.csv
Contains historical loan application data of customers
Helps analyze past loan behavior

⚙️ Technologies & Libraries Used

Python
Pandas – Data manipulation
NumPy – Numerical computations
Matplotlib – Data visualization
Seaborn – Statistical visualizations

🔍 Project Workflow
1️⃣ Data Loading

Loaded both application and previous application datasets
Inspected structure, data types, and missing values

2️⃣ Data Cleaning

Removed columns with more than 47% missing values
Handled missing values using:
Mode (categorical columns)
Median (numerical columns)
Converted negative day values to positive for better interpretation

3️⃣ Feature Engineering

Created new features such as:
YEAR_BIRTH
YEAR_EMPLOYED
AGE_CATEGORY
AMT_CREDIT_CATEGORY
Converted continuous variables into meaningful categories

4️⃣ Exploratory Data Analysis (EDA)

Univariate Analysis
Pie charts for categorical variables
Box plots for numerical variables
Bivariate Analysis
Target-based distribution comparison
Correlation Analysis
Heatmaps to identify relationships between numerical features

5️⃣ Dataset Merging

Merged current and previous loan datasets using SK_ID_CURR
Removed unnecessary FLAG columns

6️⃣ Business Insights

Used pivot tables and heatmaps to analyze:
Credit amount vs income type
Default behavior across customer types

📊 Key Visualizations

Missing value percentage analysi
Category distribution pie charts
Box plots for income and credit amount
KDE plots for target-based comparison
Correlation heatmaps
Pivot table heatmaps for business insights

🎯 Key Learnings

Proper handling of missing values significantly improves data quality
Feature engineering helps convert raw data into business-usable insights
Historical loan data is critical for understanding customer risk
EDA is an essential step before building any predictive model
