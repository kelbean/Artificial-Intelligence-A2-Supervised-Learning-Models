import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Loading dataset
df = pd.read_csv("student_habits_performance.csv")

# Selecting numeric features

X = df[
    [
        'age',
        'study_hours_per_day',
        'social_media_hours',
        'netflix_hours',
        'attendance_percentage',
        'sleep_hours',
        'exercise_frequency',
        'mental_health_rating'
    ]
]

# Selecting target
y = df['exam_score']

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state = 42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)

# Displaying results
print("Polynomial Regression Model - Performance Results")
print(f"MAE: {MAE:.2f}")
print(f"MSE: {MSE:.2f}")