# Iris Classification MLOps Project (CI/CD, Unit Testing, Automated Reporting)

This project automates the process of training and evaluating a Decision Tree Classifier on the Iris dataset.
It incorporates CI/CD for continuous integration, unit testing for model validation, and automated reporting via CML, ensuring a robust and streamlined workflow from development to deployment.

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

### Testing Scripts

#### `test.py`

- **Purpose**: Contains unit tests to validate the trained model's performance and sanity checks.
- **Functionality**:
  - Sets up test environment by locating the latest artifacts directory, model file, metrics file, and data file.
  - `test_model_metrics_sanity`: Verifies that the model's accuracy is at least 0.6 and checks metrics integrity.
  - `test_model_prediction_shape`: Loads the model and tests predictions on a sample of the data to ensure correct output shape.
- **Dependencies**: Requires `unittest`, `os`, `pandas`, `joblib`, and `scikit-learn`.
- **Usage**: Run with `python -m unittest test.py -v` to execute the tests and validate the model.

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

### CI/CD Configuration

#### `.github/workflows/sanity.yml`

- **Purpose**: Defines a GitHub Actions workflow for continuous integration, testing, and automated reporting.
- **Functionality**:
  - Triggers on pushes to `dev` and `main` branches, pull requests to `main`, and manual dispatch.
  - Sets up a Ubuntu environment with Python 3.9.
  - Installs dependencies from `requirements.txt`.
  - Pulls DVC artifacts (model and data) if available.
  - Runs unit tests using `test.py` and captures output.
  - Creates a CML report including test results and training metrics, then publishes it as a comment on the pull request or commit.
- **Dependencies**: Relies on GitHub Actions runners, DVC for data versioning, and CML for reporting.
- **Usage**: Automatically executed on specified triggers; ensures code quality and model performance through automated testing and reporting.

### Version Control and Data Management

#### `.dvcignore`

- **Purpose**: Specifies files and directories to be ignored by DVC (Data Version Control).
- **Usage**: Prevents unnecessary files from being tracked by DVC, similar to `.gitignore` for Git.

#### `.gitignore`

- **Purpose**: Specifies files and directories to be ignored by Git.
- **Usage**: Excludes sensitive or generated files (e.g., artifacts, models) from version control.

#### `artifacts.dvc`

- **Purpose**: DVC file tracking the artifacts directory for data and model versioning.
- **Usage**: Allows versioning and sharing of trained models and metrics via DVC.

#### `data/iris.csv.dvc` and `data/iris_augmented.csv.dvc`

- **Purpose**: DVC files tracking the respective data files for versioning and reproducibility.
- **Usage**: Enables tracking changes to datasets and facilitates data pipeline management.

## Pipeline Overview

1. **Data Augmentation**: Run `augment_data.py` to create an augmented dataset.
2. **Training**: Execute `train.py` with either original or augmented data to train the model.
3. **Artifacts**: Model and metrics are saved in timestamped directories under `artifacts/`.
4. **Evaluation**: Metrics include accuracy, precision, recall, and F1-score for comprehensive performance assessment.
5. **Testing**: Run `test.py` to validate model performance and sanity.
6. **CI/CD**: Push changes to trigger the GitHub Actions workflow for automated testing and reporting.

## Prerequisites

- Python 3.x
- Required libraries listed in `requirements.txt`
- Git for version control
- DVC for data versioning (optional, for artifact management)

## Running the Pipeline

1. Install dependencies: `pip install -r requirements.txt`
2. Augment data (optional): `python augment_data.py`
3. Train model: `python train.py` (uses original data) or `python train.py --data data/iris_augmented.csv` (uses augmented data)
4. Run tests: `python -m unittest test.py -v`
5. Check artifacts directory for saved model and metrics.
6. Commit and push changes to trigger CI/CD workflow.
