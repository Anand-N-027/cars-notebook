import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Suppress runtime layout and conversion warnings
warnings.filterwarnings('ignore')

# Ensure inline plotting behavior within notebook instances
# %matplotlib inline

# =====================================================================
# 1. SETUP AND REPOSITORIES DATA INGESTION
# =====================================================================
FILE_PATH = '/Users/anand/Documents/Datasets/car_price_dataset.csv'

if not os.path.exists(FILE_PATH):
    print(f"[!] Warning: Baseline dataset not found at target path: '{FILE_PATH}'")
    print("Please verify your file placement to ensure successful execution.")
else:
    # Load raw data from CSV
    df = pd.read_csv(FILE_PATH)
    print("--- Initial Raw Tabular Data Framework Preview ---")
    print(df.head())

    # =====================================================================
    # 2. DATA ENGINEERING & MULTI-COLLINEARITY HANDLING
    # =====================================================================
    # Check for missing values across vectors
    null_summary = df.isnull().sum()
    
    # Run categorical variable transformations via One-Hot Dummy encoding
    # Use drop_first=True to resolve the Dummy Variable Trap (collinearity prevention)
    df_encoded = pd.get_dummies(df, drop_first=True)
    print("\n--- Encoded Feature Structural Matrix Post-Transform Preview ---")
    print(df_encoded.head())

    # =====================================================================
    # 3. MATRIX SEGMENTATION AND SEPARATED SPLITS
    # =====================================================================
    # Isolate variables matrix from target array label
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']

    # Generate 80/20 train/test structural validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nMatrix Allocation -> X_train: {X_train.shape} | X_test: {X_test.shape}")

    # =====================================================================
    # 4. LINEAR REGRESSION TRAINING EXECUTION
    # =====================================================================
    # Initialize OLS (Ordinary Least Squares) linear architecture 
    model = LinearRegression()
    
    # Optimize network parameters using training arrays
    model.fit(X_train, y_train)
    print("\n[+] Multiple Linear Regression Model Optimization Complete.")

    # =====================================================================
    # 5. PREDICTIVE INFERENCE AND EVALUATION METRICS
    # =====================================================================
    # Run test matrix through trained framework parameters
    y_pred = model.predict(X_test)

    # Compute validation evaluation scores
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Final Model Validation Metrics Summary ---")
    print(f"Mean Squared Error (MSE) : {mse:.4f}")
    print(f"R-squared (R2) Accuracy  : {r2:.4f}")

    # =====================================================================
    # 6. RESIDUAL VISUALIZATION ANALYSIS
    # =====================================================================
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Car Price Prediction: Actual vs. Predicted Values Plot")
    plt.xlabel("Actual Car Sale Prices ($)")
    plt.ylabel("Predicted Car Sale Prices ($)")
    plt.grid(True, linestyle=":")
    plt.show()
