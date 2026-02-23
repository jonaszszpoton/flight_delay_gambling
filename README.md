
# Flight Delay Gambling

This project creates a synthetic dataset of EU flights. It also builds and evaluates a machine learning model to predict whether a flight will be delayed. Then it creates a business Monte-Carlo simulation to analyse whether it's possible to make money (from compensations) out of flight delay gambling.

## Problem Description

The primary goal is to use historical flight data to identify patterns and key drivers of delays. The project then uses these insights to build a predictive model. The analysis involves data cleaning, feature engineering, and training a Random Forest Classifier to classify flights as either "on-time" or "delayed". Combined with a known delay reason, distance and ticket cost, we can estimate how much money can we gain out of compensations for applicable delayed flights. We can then estimate the probability of net profit and do sensitivity analysis, treating flight delay gambling as a business problem. 

## Technology Stack

The project is built using Python and the following core libraries:
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For machine learning tasks (feature encoding, model training, and evaluation).
- **Matplotlib & Seaborn:** For data visualization.

## Directory Structure

The project is organized as follows:

```
/main
├── main.ipynb                    # Jupyter Notebook with detailed, step-by-step analysis.
├── main.py                       # A streamlined Python script for executing the analysis and prediction.
├── requirements.txt              # A file listing the Python dependencies for the project.
└── pipeline.svg               # Data pipeline 

/data
├─ raw_data.csv                   # The primary dataset used for the analysis.
├── macro_risk_data.csv           # Dataset after macro risk prediction (Time-series) 
├── test_scores_data.csv          # Dataset after individual RF predictions (scores) - only test rows
```

## How to Run

To run this project, follow these steps:

1.  **Prerequisites:** Ensure you have Python and `pip` installed.

2.  **Install Dependencies:** Navigate to the project directory in your terminal and install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute the Analysis:** You have two options to run the analysis:

    *   **Option A: Run the Python Script**
        For a direct execution of the data processing and model evaluation, run the script:
        ```bash
        python main.py
        ```
        This will generate data, train the models, print the model's performance metrics to the console, as well as simulating the financial outcomes based on predictions.


    *   **Option B: Use the Jupyter Notebook**
        For an interactive, step-by-step exploration of the analysis, start the Jupyter Notebook server:
        ```bash
        jupyter notebook
        ```
        Then, open the `main.ipynb` file in your browser.