# Car Price Prediction Using Machine Learning

## Overview

This project develops a machine learning pipeline for predicting used car prices based on vehicle characteristics such as manufacturing year, engine size, mileage, number of doors, owner count, fuel type, transmission type, brand, and model.

The workflow includes:

- Data loading and inspection
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and categorical encoding
- Correlation analysis
- Linear Regression model training
- Model evaluation using regression metrics
- Visualization of predictions

The goal is to understand which vehicle attributes influence car prices and build a predictive model capable of estimating market value.

---

## Dataset

The dataset contains synthetic vehicle listings with the following features:

| Feature | Description |
|----------|-------------|
| Brand | Vehicle manufacturer |
| Model | Vehicle model |
| Year | Manufacturing year |
| Engine_Size | Engine displacement |
| Fuel_Type | Petrol, Diesel, Electric, Hybrid |
| Transmission | Manual, Automatic, Semi-Automatic |
| Mileage | Distance driven |
| Doors | Number of doors |
| Owner_Count | Previous owners |
| Price | Vehicle selling price (Target Variable) |

### Example Record

| Brand | Model | Year | Engine_Size | Fuel_Type | Transmission | Mileage | Price |
|---------|---------|---------|-------------|------------|------------|---------|---------|
| Kia | Rio | 2020 | 4.2 | Diesel | Manual | 289944 | 8501 |

---

## Project Structure

```text
.
├── car_price_prediction.ipynb
├── Car_Price_Dataset.csv
├── README.md
└── outputs/
    ├── correlation_heatmap.png
    ├── pairplot.png
    └── prediction_scatterplot.png
```

---

## Technologies Used

### Programming Language

- Python 3

### Libraries

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Data Preprocessing

Several preprocessing techniques were applied before model training:

### Missing Value Check

```python
df.isnull().sum()
```

No missing values were detected in the dataset.

### Categorical Encoding

The following categorical variables were converted using one-hot encoding:

- Brand
- Model
- Fuel_Type
- Transmission

```python
pd.get_dummies()
```

### Feature Engineering

The encoded dataset transforms textual vehicle attributes into numerical format suitable for machine learning algorithms.

---

## Exploratory Data Analysis

### Correlation Heatmap

A correlation matrix was generated to analyze relationships between numerical variables.

Key observations:

- Newer vehicles generally have higher prices.
- Higher mileage tends to reduce vehicle value.
- Engine size has a moderate positive relationship with price.
- Owner count shows weak correlation with price.

### Pairplot Analysis

Pairwise visualizations were created for:

- Year
- Engine Size
- Mileage
- Price
- Fuel Type

This provides a better understanding of feature distributions and relationships.

---

## Machine Learning Model

### Algorithm

Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### Train-Test Split

Dataset partition:

```python
80% Training
20% Testing
```

Using:

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

## Model Evaluation

Performance metrics:

### Mean Squared Error (MSE)

```text
4231.92
```

### R² Score

```text
0.9995
```

The extremely high R² score indicates that the model explains nearly all variance in the target variable.

---

## Prediction Visualization

Predicted prices were compared against actual prices using a scatter plot.

```python
plt.scatter(y_test, y_pred)
```

A strong linear relationship between actual and predicted values demonstrates excellent predictive performance.

---

## Workflow

```text
Load Dataset
      ↓
Inspect Data
      ↓
Check Missing Values
      ↓
Encode Categorical Features
      ↓
Perform EDA
      ↓
Generate Correlation Analysis
      ↓
Train/Test Split
      ↓
Train Linear Regression Model
      ↓
Predict Test Data
      ↓
Evaluate Model Performance
      ↓
Visualize Results
```

---

## Running the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Launch Notebook

```bash
jupyter notebook
```

Open:

```text
car_price_prediction.ipynb
```

Run all cells.

---

## Sample Output

```text
Mean Squared Error: 4231.92
R-squared: 0.9995
```

---

## Key Insights

- Vehicle age significantly impacts price.
- Mileage negatively affects resale value.
- Larger engines generally correspond to higher prices.
- Fuel type contributes differently to pricing trends.
- Proper categorical encoding substantially improves model performance.

---

## Future Improvements

- Implement Polynomial Regression
- Compare Tree-Based Models
- Apply Random Forest Regressor
- Train XGBoost Regressor
- Hyperparameter Optimization
- Cross Validation
- Feature Importance Analysis
- Deploy as a web application using Flask or Streamlit

---

## Author

Anand Narendran

Machine Learning | Data Science | Predictive Analytics
