# Iris Classification MLOps Project with Data Augmentation

This project implements an end-to-end MLOps pipeline for training and evaluating an Iris flower classification model using Decision Tree Classifier. It includes data augmentation to enhance the dataset and improve model robustness.

## Files and Their Utilities

### Core Scripts

#### `train.py`

- **Purpose**: Handles the model training and evaluation process for the Iris classification task.
- **Functionality**:
  - Accepts a command-line argument for the data file path (defaults to "data/iris.csv").
  - Loads the specified dataset.
  - Splits the data into training and testing sets (60/40 split with stratification).
  - Trains a Decision Tree Classifier with max_depth=3 and random_state=1.
  - Evaluates the model on the test set, calculating accuracy, precision, recall, and F1-score (macro-averaged).
  - Saves evaluation metrics as a CSV file in a timestamped artifacts directory.
  - Saves the trained model as a joblib file in the same artifacts directory.
  - Prints the accuracy to the console.
- **Dependencies**: Requires `pandas`, `numpy`, `scikit-learn`, `joblib`, and `datetime`.
- **Usage**: Run with `python train.py --data <path_to_data>` to train the model on specified data.

#### `augment_data.py`

- **Purpose**: Performs data augmentation on the original Iris dataset to create a more diverse training set.
- **Functionality**:
  - Loads the original Iris dataset from `data/iris.csv`.
  - Adds Gaussian noise (mean=0, std=0.1) to all numerical features (sepal_length, sepal_width, petal_length, petal_width).
  - Saves the augmented dataset to `data/iris_augmented.csv` without modifying the species labels.
  - Prints a confirmation message upon completion.
- **Dependencies**: Requires `pandas` and `numpy`.
- **Usage**: Run with `python augment_data.py` to generate augmented data for training.

### Configuration and Dependencies

#### `requirements.txt`

- **Purpose**: Lists all Python dependencies required to run the project scripts.
- **Content**:
  - `pandas`: For data manipulation and CSV handling.
  - `numpy`: For numerical operations, including random noise generation.
  - `scikit-learn`: For machine learning algorithms and evaluation metrics.
  - `joblib`: For efficient serialization of the trained model.
- **Usage**: Install dependencies with `pip install -r requirements.txt`.

### Data Files

#### `data/iris.csv`

- **Purpose**: Contains the original Iris flower dataset used for training and evaluation.
- **Structure**: CSV file with 150 rows and 5 columns:
  - `sepal_length`: Sepal length in cm (float)
  - `sepal_width`: Sepal width in cm (float)
  - `petal_length`: Petal length in cm (float)
  - `petal_width`: Petal width in cm (float)
  - `species`: Target variable with three classes: setosa, versicolor, virginica (string)
- **Source**: Classic Iris dataset by R.A. Fisher.
- **Usage**: Used by `train.py` for model training and by `augment_data.py` as input for augmentation.

#### `data/iris_augmented.csv`

- **Purpose**: Contains the augmented version of the Iris dataset with added noise for improved model generalization.
- **Structure**: Same as `iris.csv` but with slight perturbations added to numerical features.
- **Generation**: Created by `augment_data.py` from the original dataset.
- **Usage**: Can be used as an alternative or additional training dataset to improve model robustness against small variations in input data.

## Pipeline Overview

1. **Data Augmentation**: Run `augment_data.py` to create an augmented dataset.
2. **Training**: Execute `train.py` with either original or augmented data to train the model.
3. **Artifacts**: Model and metrics are saved in timestamped directories under `artifacts/`.
4. **Evaluation**: Metrics include accuracy, precision, recall, and F1-score for comprehensive performance assessment.

## Prerequisites

- Python 3.x
- Required libraries listed in `requirements.txt`

## Running the Pipeline

1. Install dependencies: `pip install -r requirements.txt`
2. Augment data (optional): `python augment_data.py`
3. Train model: `python train.py` (uses original data) or `python train.py --data data/iris_augmented.csv` (uses augmented data)
4. Check artifacts directory for saved model and metrics.
