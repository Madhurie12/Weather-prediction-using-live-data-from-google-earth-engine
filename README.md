# Weather Prediction using Live Data from Google Earth Engine

This project focuses on predicting weather conditions using live data obtained from Google Earth Engine. It utilizes the Seattle weather dataset and employs a Random Forest Classifier for prediction.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Data Preprocessing](#data-preprocessing)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Converting the Estimator](#converting-the-estimator)
- [Earth Engine Classification](#earth-engine-classification)
- [Image Classification](#image-classification)
- [Results Visualization](#results-visualization)
- [Usage](#usage)

## Introduction

This project aims to predict weather conditions based on live data obtained from Google Earth Engine. It utilizes the Seattle weather dataset, which contains information about precipitation, temperature, wind speed, and weather conditions. By employing a Random Forest Classifier, the project predicts the weather condition based on the given input features.

## Dataset

The project uses the "seattle-weather updated.csv" dataset, which contains the following columns:

- Date: The date of the weather data
- Precipitation: Amount of precipitation
- Temp_max: Maximum temperature
- Temp_min: Minimum temperature
- Wind: Wind speed
- Weather: Weather condition (label)

## Implementation

The project is implemented in Jupyter Notebook using Python and the following libraries:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import re
import missingno as mso 
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import ml
import ee
```

## Data Preprocessing

The dataset is preprocessed to handle missing values and perform exploratory data analysis. The `missingno` library is used to check for missing values, while statistical analysis and data visualization are performed using `scipy.stats`, `matplotlib.pyplot`, and `seaborn`.

```python
# Check for missing values
mso.matrix(data)

# Perform statistical analysis
stats.describe(data)

# Data visualization
sns.pairplot(data)
```

## Data Preparation

The dataset is prepared for model training by dropping the "date" column and defining the feature names and the target label.

```python
data = data.drop(columns='date', axis=1)
feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind']
label = "weather"
X = data[feature_names]
y = data[label]
```

## Model Training

A Random Forest Classifier is trained using the prepared dataset. The number of trees is set to 100, and the classifier is fitted using the `ensemble.RandomForestClassifier` class from the `sklearn.ensemble` module.

```python
n_trees = 100
rf = ensemble.RandomForestClassifier(n_trees).fit(X, y)
```

## Converting the Estimator

The trained Random Forest Classifier is converted into a list of strings using the `rf_to_strings` function from the `ml` library.

```python
trees = ml.rf_to_strings(rf, feature_names)
```

## Earth Engine Classification

The list of strings is used to create an Earth Engine classifier using the `strings

_to_classifier` function from the `ml` library.

```python
ee_classifier = ml.strings_to_classifier(trees)
ee_classifier.getInfo()
```

## Image Classification

A Landsat 8 TOA composite image is obtained using the Earth Engine API. The image is classified using the Earth Engine classifier created from the local training.

```python
l8 = ee.ImageCollection("LANDSAT/LC09/C02/T1")

image = ee.Algorithms.Landsat.simpleComposite(
   collection=l8.filterDate('2021-10-31', '2023-03-18'), 
   asFloat=True
)

classified = image.select(feature_names).classify(ee_classifier)
```

## Results Visualization

The results are visualized using the Earth Engine API and displayed on a map. The true color image of the classified area is obtained using the Landsat data.

```python
dataset = ee.ImageCollection('LANDSAT/LC09/C02/T1').filterDate('2022-01-01', '2022-02-01')
trueColor432 = dataset.select(['B4', 'B3', 'B2'])
trueColor432Vis = {'min': 0.0, 'max': 30000.0}
Map.setCenter(6.746, 46.529, 6)
Map.addLayer(trueColor432, trueColor432Vis, 'True Color (432)')
```

## Usage

To use this project, follow these steps:

1. Install the required libraries listed in the implementation section.
2. Download the "seattle-weather updated.csv" dataset and place it in the appropriate directory.
3. Open the Jupyter Notebook file and run the code cells sequentially.

Please note that a valid Google Earth Engine API account and credentials are required to access the Earth Engine data.

That's it! You can now explore and utilize this Jupyter Notebook project for weather prediction using live data from Google Earth Engine.

Feel free to modify the code and experiment with different models or datasets to enhance the weather prediction capabilities.
