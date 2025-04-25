# Iris Flower Classification Project

![Iris Species](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/220px-Iris_virginica.jpg)

A comprehensive machine learning project to classify Iris flower species using various algorithms with detailed data analysis and model evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Algorithms Used](#algorithms-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

## Project Overview
This project demonstrates:
- Complete EDA process
- Outlier detection and handling
- Multiple classification algorithms
- Model evaluation with cross-validation
- Comprehensive performance metrics

## Dataset
**IRIS.csv** (included in repository):
- 150 samples (50 per class)
- Features:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- Target classes:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

## Data Preprocessing
1. **Outlier Handling**:
   - Detected outliers in Sepal Width using boxplots
   - Applied Winsorization (IQR method):
     ```python
     q1 = df['sepal_width'].quantile(0.25)
     q3 = df['sepal_width'].quantile(0.75)
     iqr = q3 - q1
     upper_limit = q3 + (1.5*iqr)
     lower_limit = q1 - (1.5*iqr)
     df['sepal_width'] = np.where(df['sepal_width']>upper_limit, upper_limit, 
                        np.where(df['sepal_width']<lower_limit, lower_limit, df['sepal_width']))
     ```

2. **Data Splitting**:
   - 80-20 train-test split
   - Random state = 3 for reproducibility

## Algorithms Used
1. **Decision Tree Classifier**
2. **K-Nearest Neighbors (KNN)** (k=9)
3. **Random Forest Classifier** (25 estimators)

## Results
### Model Performance
| Algorithm          | Accuracy | Cross-Val Score (10-fold) |
|--------------------|----------|---------------------------|
| Decision Tree      | 100%     | 94.67%                    |
| KNN                | 100%     | 97.33%                    |
| Random Forest      | 100%     | 95.33%                    |

### Confusion Matrix (All Models)




### Key Metrics (All Models)
- Precision: 100%
- Recall: 100% 
- F1-Score: 100%

## Installation
1. Clone repository:
   ```bash
   git clone https://github.com/your-username/iris-classification.git
   cd iris-classification

## Install requirements
 pip install -r requirements.txt

 ## requirements.txt
      numpy>=1.21.0
      pandas>=1.3.0
      scikit-learn>=1.0
      matplotlib>=3.4.0
      seaborn>=0.11.0
  ## Usage
    jupyter notebook Iris_Classification_Analysis.ipynb

  ## Future Work 
      Implement data augmentation for better generalisation
      Add neural network implementation
      Create a Flask API for predictions
      Develop a comparative visualisation dashboard


      
