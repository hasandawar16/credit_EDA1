import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------

# 📦 Importing Libraries

# -----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display full columns in pandas output
pd.set_option("display.max_columns", None)

# -----------------------------------------------

# 📂 Load the Application Data

# -----------------------------------------------

app_data = pd.read_csv("D:\\document\\application_data.csv")

print("✅ Dataset loaded successfully!")
print(app_data.info())

# -----------------------------------------------

# 🧹 Data Cleaning & Handling Missing Values

# -----------------------------------------------

print("\nPercentage of missing values in each column:")
print(app_data.isnull().mean() * 100)

# Drop columns having more than 47% null values
threshold = int(((100 - 47) / 100) * app_data.shape[0])
app_df = app_data.dropna(axis=1, thresh=threshold)

print("\nAfter dropping high-null columns:")
print(app_df.shape)

# Fill missing values in OCCUPATION_TYPE
print("\nBefore filling OCCUPATION_TYPE:", app_df.OCCUPATION_TYPE.isnull().mean() * 100, "%")
app_df["OCCUPATION_TYPE"].fillna("Others", inplace=True)
print("After filling OCCUPATION_TYPE:", app_df.OCCUPATION_TYPE.isnull().mean() * 100, "%")

# Fill missing values in EXT_SOURCE_3 with median
app_df["EXT_SOURCE_3"].fillna(app_df["EXT_SOURCE_3"].median(), inplace=True)

# Fill few numeric columns with mode
num_mode_cols = [
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"
]

for col in num_mode_cols:
    app_df[col].fillna(app_df[col].mode()[0], inplace=True)

# Fill low-null columns appropriately
fill_cols = {
    "NAME_TYPE_SUITE": "mode",
    "CNT_FAM_MEMBERS": "mode",
    "EXT_SOURCE_2": "median",
    "AMT_GOODS_PRICE": "median",
    "AMT_ANNUITY": "median",
    "OBS_30_CNT_SOCIAL_CIRCLE": "median",
    "DEF_30_CNT_SOCIAL_CIRCLE": "median",
    "OBS_60_CNT_SOCIAL_CIRCLE": "median",
    "DEF_60_CNT_SOCIAL_CIRCLE": "median",
    "DAYS_LAST_PHONE_CHANGE": "median"
}

for col, method in fill_cols.items():
    if method == "mode":
        app_df[col].fillna(app_df[col].mode()[0], inplace=True)
    else:
        app_df[col].fillna(app_df[col].median(), inplace=True)

print("\nMissing values after imputation:")
print(app_df.isnull().mean() * 100)

# -----------------------------------------------

# 🧮 Convert Negative Day Values to Positive

# -----------------------------------------------

for col in ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]:
    app_df[col] = app_df[col].abs()

# -----------------------------------------------

# 🗂️ Feature Engineering

# -----------------------------------------------

app_df["YEAR_BIRTH"] = (app_df["DAYS_BIRTH"] // 365).astype(int)
app_df["YEAR_EMPLOYED"] = (app_df["DAYS_EMPLOYED"] // 365).astype(int)

app_df["AMT_CREDIT_CATEGORY"] = pd.cut(
    app_df["AMT_CREDIT"],
    bins=[0, 200000, 400000, 600000, 800000, 1000000, float("inf")],
    labels=["very low", "low", "medium", "high", "very high", "extreme"]
)

app_df["AGE_CATEGORY"] = pd.cut(
    app_df["YEAR_BIRTH"],
    bins=[0, 25, 45, 65, 85],
    labels=["below 25", "25-45", "45-65", "65-85"]
)

# -----------------------------------------------

# 📊 Univariate Analysis

# -----------------------------------------------

cat_cols = list(app_df.select_dtypes(include="object").columns)
num_cols = list(app_df.select_dtypes(include=["int64", "float64"]).columns)

print("\nCategorical Columns:", cat_cols)
print("\nNumerical Columns:", num_cols)

# Pie chart for categorical variables
for col in cat_cols[:3]:  # showing only first 3 for simplicity
    print(f"\nDistribution of {col}:")
    print(app_df[col].value_counts(normalize=True) * 100)
    plt.figure(figsize=(5, 5))
    app_df[col].value_counts(normalize=True).plot.pie(autopct="%1.2f%%")
    plt.title(col)
    plt.show()

# Boxplot for few numerical columns
for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_GOODS_PRICE"]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=app_df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

# -----------------------------------------------

# 🔍 Target-based Analysis

# -----------------------------------------------

tar_0 = app_df[app_df.TARGET == 0]
tar_1 = app_df[app_df.TARGET == 1]

plt.figure(figsize=(8, 5))
sns.kdeplot(tar_0["AMT_GOODS_PRICE"], label="Target 0")
sns.kdeplot(tar_1["AMT_GOODS_PRICE"], label="Target 1")
plt.legend()
plt.title("Density Plot of AMT_GOODS_PRICE by Target")
plt.show()

# -----------------------------------------------

# 🔗 Correlation Heatmap

# -----------------------------------------------

corr_cols = ["AMT_INCOME_TOTAL", "YEAR_EMPLOYED", "YEAR_BIRTH", "AMT_CREDIT"]
corr_data = app_df[corr_cols]

plt.figure(figsize=(8, 6))
sns.heatmap(corr_data.corr(), annot=True, cmap="RdYlGn")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------------------------

# 📂 Load & Clean Previous Application Data

# -----------------------------------------------

papp_data = pd.read_csv("D:\\document\\previous_application.csv")

print("\nPrevious Application Data Info:")
print(papp_data.info())

threshold_prev = int(((100 - 49) / 100) * papp_data.shape[0])
papp_df = papp_data.dropna(axis=1, thresh=threshold_prev)

for col in papp_df.select_dtypes(include=["int64", "float64"]).columns:
    papp_df[col] = papp_df[col].abs()

papp_df["AMT_CREDIT_CATEGORY"] = pd.cut(
    papp_df["AMT_CREDIT"],
    bins=[0, 200000, 400000, 600000, 800000, 1000000, float("inf")],
    labels=["very low", "low", "medium", "high", "very high", "extreme"]
)

if "YEAR_BIRTH" in papp_df.columns:
    papp_df["AGE_CATEGORY"] = pd.cut(
        papp_df["YEAR_BIRTH"],
        bins=[0, 25, 45, 65, 85],
        labels=["below 25", "25-45", "45-65", "65-85"]
    )

# -----------------------------------------------

# 🧩 Merge Both Datasets

# -----------------------------------------------

merge_df = app_df.merge(papp_df, on="SK_ID_CURR", how="left")
merge_df.drop(columns=[col for col in merge_df.columns if col.startswith("FLAG")], inplace=True, errors="ignore")

# -----------------------------------------------

# 🧠 Pivot Table & Heatmap

# -----------------------------------------------

if {"NAME_INCOME_TYPE", "NAME_CLIENT_TYPE", "TARGET", "AMT_CREDIT"}.issubset(merge_df.columns):
    res1 = pd.pivot_table(
        merge_df,
        index=["NAME_INCOME_TYPE", "NAME_CLIENT_TYPE"],
        columns="TARGET",
        values="AMT_CREDIT",
        aggfunc="mean"
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(res1, annot=True, cmap="BuPu")
    plt.title("Pivot Table Heatmap")
    plt.show()

print("\n✅ Final merged dataset ready for analysis!")
print(merge_df.info())