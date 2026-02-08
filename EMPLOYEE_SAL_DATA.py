# Employee Salary Prediction
# Author: Md Asif

# --------- Import Libraries ---------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --------- Create Sample Dataset ---------
data = {
    "Age": [25, 30, 45, 28, 35, 40, 50, 29, 33, 38],
    "Education": ["Bachelors", "Masters", "PhD", "Bachelors", "Masters",
                  "PhD", "Masters", "Bachelors", "Masters", "PhD"],
    "JobRole": ["Developer", "Manager", "Scientist", "Analyst", "Developer",
                "Manager", "Scientist", "Analyst", "Developer", "Manager"],
    "Department": ["IT", "HR", "R&D", "Finance", "IT",
                   "HR", "R&D", "Finance", "IT", "HR"],
    "YearsExperience": [2, 6, 15, 3, 8, 12, 20, 4, 7, 10],
    "Salary": [40000, 70000, 120000, 45000, 80000,
               100000, 150000, 48000, 75000, 95000]
}

df = pd.DataFrame(data)

# --------- Data Info ---------
print("\nDataset Preview:\n")
print(df.head())

# --------- Encode Categorical Columns ---------
le = LabelEncoder()

df["Education"] = le.fit_transform(df["Education"])
df["JobRole"] = le.fit_transform(df["JobRole"])
df["Department"] = le.fit_transform(df["Department"])

# --------- Feature Selection ---------
X = df.drop("Salary", axis=1)
y = df["Salary"]

# --------- Train-Test Split ---------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------- Linear Regression Model ---------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\nLinear Regression Results:")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))

# --------- Random Forest Model ---------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results:")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))

# --------- Model Comparison ---------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_rf)
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred_lr),
        mean_absolute_error(y_test, y_pred_rf)
    ]
})

print("\nModel Comparison:\n")
print(results)

# --------- Visualization ---------
plt.figure()
sns.scatterplot(x=df["YearsExperience"], y=df["Salary"])
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# --------- Salary Prediction for New Employee ---------
new_employee = np.array([[32, 1, 2, 0, 6]])
predicted_salary = rf.predict(new_employee)

print("\nPredicted Salary for New Employee:", predicted_salary[0])
