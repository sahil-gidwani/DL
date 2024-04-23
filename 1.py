import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %matplotlib inline

df = pd.read_csv("https://raw.githubusercontent.com/sahil-gidwani/DL/main/data/BostonHousing.csv")

df.head()

# CRIM: Per capita crime rate by town
# ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
# INDUS: Proportion of non-retail business acres per town
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX: Nitric oxide concentration (parts per 10 million)
# RM: Average number of rooms per dwelling
# AGE: Proportion of owner-occupied units built prior to 1940
# DIS: Weighted distances to five Boston employment centers
# RAD: Index of accessibility to radial highways
# TAX: Full-value property tax rate per $10,000
# PTRATIO: Pupil-teacher ratio by town
# B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
# LSTAT: Percentage of lower status of the population
# MEDV: Median value of owner-occupied homes in $1000s ------ TARGET

df.isnull().sum()

df.dropna(inplace = True)

df.info()

correlation_matrix = df.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

# sns.pairplot(df)

X = df.drop('medv', axis = 1)
Y = df['medv']

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# lin_model = LinearRegression()
# lin_model.fit(X_train, Y_train)

# Define the number of features
num_features = X_train.shape[1]

# Create a Sequential model
model = Sequential()

# Add multiple dense layers
model.add(Dense(64, input_dim=num_features, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Add the final output layer with linear activation for linear regression
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

# model evaluation for training set
y_train_predict = model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
mse = mean_squared_error(Y_train, y_train_predict)
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('Mean Squared Error (MSE) is {}'.format(mse))
print('Root Mean Squared Error (RMSE) is {}'.format(rmse))
print('R2 score is {}'.format(r2))

# model evaluation for testing set
y_test_predict = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
mse = mean_squared_error(Y_test, y_test_predict)
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('Mean Squared Error (MSE) is {}'.format(mse))
print('Root Mean Squared Error (RMSE) is {}'.format(rmse))
print('R2 score is {}'.format(r2))

"""
Linear regression is a fundamental statistical method used for modeling the relationship between a dependent variable (target) and one or more independent variables (features). It assumes a linear relationship between the input variables and the target variable. The goal of linear regression is to find the best-fitting straight line that describes the relationship between the independent variables and the dependent variable.

### Mathematical Representation:
In simple linear regression, where there is only one independent variable, the relationship between the independent variable \(X\) and the dependent variable \(Y\) can be represented as:

\[ Y = \beta_0 + \beta_1X + \varepsilon \]

- \( Y \) is the dependent variable (target).
- \( X \) is the independent variable (feature).
- \( \beta_0 \) is the y-intercept (bias).
- \( \beta_1 \) is the coefficient for \( X \).
- \( \varepsilon \) is the error term, representing the difference between the predicted and actual values.

The linear regression model aims to estimate the values of the coefficients \( \beta_0 \) and \( \beta_1 \) that minimize the sum of squared differences between the observed and predicted values of \( Y \).

### Assumptions:
1. **Linearity**: Assumes a linear relationship between the independent and dependent variables.
2. **Independence**: Assumes that the observations are independent of each other.
3. **Homoscedasticity**: Assumes that the variance of the errors is constant across all levels of the independent variable.
4. **Normality of Errors**: Assumes that the errors are normally distributed.
5. **No Multicollinearity**: Assumes that the independent variables are not highly correlated with each other.

### Steps in Linear Regression:
1. **Data Collection**: Gather data on the independent and dependent variables.
2. **Data Preprocessing**: Clean the data, handle missing values, and perform feature scaling if necessary.
3. **Model Training**: Use the training data to fit the linear regression model to the data. This involves estimating the coefficients \( \beta_0 \) and \( \beta_1 \) that minimize the error term.
4. **Model Evaluation**: Assess the performance of the model using evaluation metrics such as mean squared error (MSE), root mean squared error (RMSE), and \( R^2 \) score.
5. **Prediction**: Once the model is trained and evaluated, use it to make predictions on new or unseen data.

### Types of Linear Regression:
1. **Simple Linear Regression**: Involves one independent variable.
2. **Multiple Linear Regression**: Involves two or more independent variables.
3. **Polynomial Regression**: Allows for non-linear relationships by adding polynomial terms to the model.
4. **Ridge Regression**: Adds a penalty term to the loss function to prevent overfitting.
5. **Lasso Regression**: Similar to ridge regression but uses the L1 penalty term, leading to sparsity in the coefficients.

Linear regression is widely used in various fields such as economics, finance, engineering, and social sciences for predicting outcomes, understanding relationships between variables, and making decisions based on data.

Linear regression models can be evaluated using various metrics to assess their performance and the quality of predictions. Here are some commonly used evaluation metrics for linear regression:

1. **Mean Squared Error (MSE)**:
   - MSE calculates the average of the squares of the errors between the actual and predicted values. It penalizes larger errors more heavily than smaller ones.
   - Formula: \( MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \)
   - Where \( n \) is the number of observations, \( y_i \) is the actual value, and \( \hat{y}_i \) is the predicted value for observation \( i \).

2. **Root Mean Squared Error (RMSE)**:
   - RMSE is the square root of the MSE and represents the average magnitude of the errors in the predicted values.
   - Formula: \( RMSE = \sqrt{MSE} \)

3. **Mean Absolute Error (MAE)**:
   - MAE calculates the average of the absolute differences between the actual and predicted values. It provides a measure of the average magnitude of errors.
   - Formula: \( MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \)

4. **R-squared (R2) Score**:
   - R2 score represents the proportion of the variance in the dependent variable that is explained by the independent variables. It ranges from 0 to 1, with 1 indicating a perfect fit.
   - Formula: \( R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \)
   - Where \( SS_{res} \) is the sum of squared residuals (errors) and \( SS_{tot} \) is the total sum of squares.

5. **Adjusted R-squared (Adjusted R2)**:
   - Adjusted R2 adjusts the R2 score to account for the number of predictors in the model. It penalizes the addition of unnecessary variables that do not improve the model's performance.
   - Formula: \( Adjusted \ R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1} \)
   - Where \( n \) is the number of observations and \( p \) is the number of predictors.

These metrics help assess the accuracy, precision, and goodness-of-fit of the linear regression model. Depending on the specific problem and context, one or more of these metrics may be used for model evaluation.
"""
