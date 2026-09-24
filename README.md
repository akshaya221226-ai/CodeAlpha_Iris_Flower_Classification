# CodeAlpha Iris Flower Classification

## Project Overview

This project performs Iris flower classification using Python and Machine Learning. The model classifies Iris flowers into three species based on their sepal and petal measurements.

## Objectives

- Analyze the Iris flower dataset.
- Clean and prepare the dataset.
- Train a Machine Learning classification model.
- Predict different Iris flower species.
- Evaluate the model performance.
- Visualize the dataset and classification results.

## Dataset

The project uses the Iris dataset containing 150 flower samples.

### Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target Classes

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Model

The project uses **Logistic Regression** for classification.

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The model is trained using the four flower measurement features.

## Model Performance

The Logistic Regression model achieved:

**Accuracy: 96.67%**

The model correctly classified 29 out of 30 test samples.

## Visualizations

The project generates five visualizations:

1. Iris Species Distribution
2. Sepal Length vs Sepal Width
3. Petal Length vs Petal Width
4. Confusion Matrix
5. Iris Feature Distribution

## Project Files

- `Iris.csv` - Original Iris dataset
- `iris_classification.py` - Main Machine Learning program
- `01_species_distribution.png` - Species distribution graph
- `02_sepal_length_vs_width.png` - Sepal comparison graph
- `03_petal_length_vs_width.png` - Petal comparison graph
- `04_confusion_matrix.png` - Model confusion matrix
- `05_feature_distribution.png` - Feature distribution graph
- `iris_predictions.csv` - Actual and predicted results
- `iris_classification_report.txt` - Classification analysis report

## Sample Predictions

The trained model successfully predicted:

- Sample 1 → Iris-setosa
- Sample 2 → Iris-versicolor
- Sample 3 → Iris-virginica

## Conclusion

The Iris Flower Classification project successfully demonstrates how Machine Learning can be used to classify flowers based on their physical measurements.

The Logistic Regression model achieved an accuracy of **96.67%**, showing strong classification performance on the test dataset.

## Internship

This project was completed as part of my **CodeAlpha Internship**.
