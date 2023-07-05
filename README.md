# Weather Prediction using Live Data from Google Earth Engine

This project focuses on predicting weather conditions using live data obtained from Google Earth Engine. It utilizes the Seattle weather dataset and employs a Random Forest Classifier for prediction.

## Dataset

The project uses the "seattle-weather updated.csv" dataset, which contains the following columns:

- Date: The date of the weather data
- Precipitation: Amount of precipitation
- Temp_max: Maximum temperature
- Temp_min: Minimum temperature
- Wind: Wind speed
- Weather: Weather condition (label)

## Implementation

1. Data Preprocessing:

   - Importing necessary libraries:
     - matplotlib.pyplot
     - seaborn
     - pandas
     - scipy
     - re
     - missingno
     - sklearn.preprocessing
     - sklearn.model_selection
     - sklearn.neighbors
     - sklearn.svm
     - sklearn.ensemble
     - xgboost
     - sklearn.metrics

   - Loading the dataset:
     ```
     data = pd.read_csv("C:/Users/deime/Downloads/seattle-weather updated.csv")
     ```

   - Exploratory Data Analysis:
     - Checking missing values using `missingno` library
     - Statistical analysis using `scipy.stats`
     - Data visualization using `matplotlib.pyplot` and `seaborn`

2. Data Preparation:

   - Dropping the "date" column:
     ```
     data = data.drop(columns='date', axis=1)
     ```

   - Defining feature names and the target label:
     ```
     feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind']
     label = "weather"
     ```

3. Model Training:

   - Random Forest Classifier:
     - Defining the number of trees:
       ```
       n_trees = 100
       ```

     - Fitting the Random Forest Classifier:
       ```
       rf = ensemble.RandomForestClassifier(n_trees).fit(X, y)
       ```

4. Converting the Estimator into a List of Strings:

   - Importing the necessary library:
     ```
     import ml
     ```

   - Converting the estimator into a list of strings:
     ```
     trees = ml.rf_to_strings(rf, feature_names)
     ```

5. Earth Engine Classification:

   - Importing the necessary Earth Engine libraries:
     ```
     import ee
     ```

   - Creating an Earth Engine classifier using the tree strings:
     ```
     ee_classifier = ml.strings_to_classifier(trees)
     ```

6. Image Classification:

   - Fetching the Landsat 8 TOA composite image using Earth Engine:
     ```
     l8 = ee.ImageCollection("LANDSAT/LC09/C02/T1")
     image = ee.Algorithms.Landsat.simpleComposite(collection=l8.filterDate('2021-10-31', '2023-03-18'), asFloat=True)
     ```

   - Classifying the image using the classifier:
     ```
     classified = image.select(feature_names).classify(ee_classifier)
     ```

7. Results Visualization:

   - Displaying the classified image using Earth Engine and Map visualization tools.

Note: Please make sure to replace the file path in the `pd.read_csv` function with the correct path to the "seattle-weather updated.csv" file in your system.
