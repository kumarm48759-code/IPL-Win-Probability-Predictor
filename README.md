## IPL Win Probability Predictor 

## Overview
The IPL Win Probability Predictor is a machine learning application that estimates the winning probability of an IPL team during a live match situation using historical IPL data.

The project applies Logistic Regression to analyze match conditions and predict the probability of a team winning or losing.

## Machine Learning Algorithm
**Logistic Regression**

Logistic Regression is a supervised machine learning algorithm used for binary classification. In this project, it estimates the probability of winning based on match-related features.

## Features
- Predicts the winning probability of IPL teams.
- Analyzes historical IPL ball-by-ball delivery data.
- Performs feature engineering to extract meaningful match statistics.
- Calculates Current Run Rate (CRR) and Required Run Rate (RRR).
- Considers runs remaining, wickets remaining, and balls remaining.
- Provides an interactive prediction interface using Streamlit.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Jupyter Notebook
- Streamlit

## Dataset
The project uses historical IPL datasets:
- `deliveries_updated_ipl_upto_2025.csv`
- `matches_updated_ipl_upto_2025.csv`

## Project Structure
```text
IPL-Win-Probability-Predictor/
│
├── ipl_win_probability.ipynb
├── ipl_app.py
├── ipl_model.pkl
├── deliveries_updated_ipl_upto_2025.csv
├── matches_updated_ipl_upto_2025.csv
├── README.md
└── .gitignore
```

## Installation and Execution

Install the required libraries:

```bash
pip install pandas numpy scikit-learn streamlit
```

Run the application:

```bash
streamlit run ipl_app.py
```

## Workflow
1. Data collection and loading.
2. Data preprocessing.
3. Feature engineering.
4. Model training using Logistic Regression.
5. Win probability prediction.
6. Deployment using Streamlit.

## Author
Machine Learning Project
