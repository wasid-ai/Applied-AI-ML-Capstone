# Applied AI & ML Capstone Project – Part 1

## Dataset

Adult Census Income Dataset

## Objective

The objective of this project is to perform data acquisition, data cleaning, exploratory data analysis (EDA), and preprocessing on the Adult Census Income dataset. This prepares the data for machine learning tasks in later parts of the capstone.

## Dataset Description

* Dataset Name: Adult Census Income Dataset
* Source: UCI Machine Learning Repository / Kaggle
* Target Variable: Income
* Total Features: 15

## Tasks Performed

### 1. Data Loading

The dataset was loaded into Python using Pandas and inspected using head(), shape(), info(), and describe().

### 2. Missing Value Analysis

Missing values were identified using isnull(). Their percentage was calculated to understand data quality.

### 3. Duplicate Detection

Duplicate rows were checked using duplicated() and removed using drop_duplicates().

### 4. Data Type Verification

All column data types were verified using dtypes.

### 5. Descriptive Statistics

Summary statistics including mean, standard deviation, minimum, maximum, and quartiles were generated using describe().

### 6. Skewness Analysis

Skewness of numerical features was calculated to understand feature distributions.

### 7. Outlier Detection

Outliers were identified using the Interquartile Range (IQR) method.

### 8. Data Visualization

The following visualizations were created:

* Histogram
* Box Plot
* Scatter Plot
* Bar Plot
* Count Plot
* Correlation Heatmap

### 9. Correlation Analysis

Pearson and Spearman correlations were analyzed to understand relationships among numerical variables.

### 10. Group Aggregation

Grouped statistics (mean, standard deviation, and count) were computed based on the Income category.

### 11. Save Cleaned Dataset

The cleaned dataset was saved as **cleaned_adult.csv**.

## Conclusion

The dataset was successfully cleaned and explored. Missing values, duplicates, outliers, and feature relationships were analyzed. The cleaned dataset is now ready for feature engineering and machine learning in Part 2.

