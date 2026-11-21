# Student Performance Prediction App

A machine learning web application built with **Python, Scikit-learn, and Streamlit** that predicts a student's performance category based on their learning activity data.

## Overview

The Student Performance Prediction App uses a Random Forest Classifier to analyze student activity indicators and predict whether a student belongs to the **High** or **Low** performance category.

The application provides a simple and interactive interface where users can enter student activity values and receive a prediction along with the model's confidence scores.

## Features

- Predicts student performance category (High or Low).
- Accepts eight student activity indicators.
- Uses a trained Random Forest Classification model.
- Displays prediction results and model confidence.
- Provides a simple, user-friendly web interface.
- Deployed using Streamlit Community Cloud.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data handling and preprocessing |
| Scikit-learn | Machine learning model development |
| Joblib | Saving and loading the trained model |
| Streamlit | Web application development |

## Dataset

The application uses the `augmented_balanced_dataset_m.csv` dataset.

It contains the following eight input features:

| Feature | Description |
|---|---|
| Submission.Actions | Number of submission-related activities |
| Quiz.Actions | Number of quiz-related activities |
| Forum.Actions | Number of forum-related activities |
| Create.Actions | Number of content creation activities |
| Read.Actions | Number of reading activities |
| Update.Actions | Number of update activities |
| Delete.Actions | Number of deletion activities |
| Days.Logged.In | Number of days logged into the learning platform |

**Target Variable:** `rating`

The model predicts one of two categories:
- **High**
- **Low**

*Note: These are the target labels in the dataset. They should not be interpreted as Pass or Fail unless their meaning is confirmed by the dataset documentation.*

## Machine Learning Model

The project uses a **Random Forest Classifier**.

### Model Configuration

- Algorithm: Random Forest Classifier
- Number of estimators: 200
- Random state: 42
- Class weight: Balanced
- Train-test split: 80% training and 20% testing
- Missing value handling: SimpleImputer using the median strategy

### Model Evaluation

The model was evaluated using a held-out test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 76.47% |
| Precision (High) | 0.84 |
| Recall (High) | 0.78 |
| F1-score (High) | 0.81 |
| Precision (Low) | 0.66 |
| Recall (Low) | 0.74 |
| F1-score (Low) | 0.70 |

The model achieved an accuracy of **76.47%** on the test dataset.

## Project Structure

```text
student_prediction_app/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── augmented_balanced_dataset_m.csv
│
├── app.py
├── requirements.txt
├── student_model.pkl
├── train_model.py
└── README.md
```

## Installation and Setup

To run the application locally:

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd student_prediction_app
```

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Dependencies

The project uses the following Python packages:

```text
streamlit==1.67.0
pandas==3.0.5
scikit-learn==1.9.1
joblib==1.6.0
```

## Deployment

The application is deployed using **Streamlit Community Cloud**.

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- The model's predictions are estimates and should not be treated as definitive judgments of student ability.
- The current model predicts High or Low ratings, not actual academic passing or failing.
- Model performance may vary when applied to data from different learning environments.

## Author

Developed as part of **Machine Problem 3 (MP3) for the course Applied Data Science CSDS 312**.

## License

This project was developed for academic purposes.
