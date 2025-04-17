import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('Life Expectancy Data.csv')

# Standardize column titles (remove extra spaces and special characters)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '')

# Check for duplicate rows
print(f"Initial duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)}")

# Check for missing values
print("\nMissing values before imputation:")
print(df.isnull().sum())

# Create a copy of the dataframe for imputation
df_imputed = df.copy()

# Separate numerical and categorical columns
numerical_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_imputed.select_dtypes(exclude=np.number).columns.tolist()

# Handle numerical columns - fill with median
num_imputer = SimpleImputer(strategy='median')
df_imputed[numerical_cols] = num_imputer.fit_transform(df_imputed[numerical_cols])

# Handle categorical columns - fill with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
df_imputed[categorical_cols] = cat_imputer.fit_transform(df_imputed[categorical_cols])

# Verify missing values after imputation
print("\nMissing values after imputation:")
print(df_imputed.isnull().sum())

# Additional cleaning - fix potential data entry issues
# For example, some columns might have 0s that should be treated as missing
# Let's identify columns where 0 might not be a valid value
potential_zero_issues = ['life_expectancy', 'adult_mortality', 'schooling']
for col in potential_zero_issues:
    if col in df_imputed.columns:
        # Replace 0 with median (for numerical columns)
        if col in numerical_cols:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].replace(0, median_val)

# Standardize country names (capitalize first letter)
if 'country' in df_imputed.columns:
    df_imputed['country'] = df_imputed['country'].str.title()

# Standardize status values
if 'status' in df_imputed.columns:
    df_imputed['status'] = df_imputed['status'].str.title()

# Save the cleaned data
df_imputed.to_csv('cleaned_life_expectancy_data.csv', index=False)

print("\nData cleaning and transformation complete!")
print(f"Final dataset shape: {df_imputed.shape}")

#This marks the end of cleaning of the data using python.


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("cleaned_life_expectancy_data.csv")

# Encode categorical columns
le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])  # Developed = 0, Developing = 1

# Define features and target
X = df.drop(columns=['life_expectancy', 'country'])  # Drop target + country (too many categories)
y = df['life_expectancy']

# Feature Scaling (Obtained the Scaler.pkl file.)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models (Test provides with R2 Score used to select the best model for the project.)
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Train, Predict, Evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append({'Model': name, 'R2 Score': r2, 'RMSE': rmse})

# Results DataFrame
results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False)
print(results_df)

# Plotting
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Model', y='R2 Score')
plt.title("Model Comparison (R2 Score)")
plt.tight_layout()
plt.show() 


import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned_life_expectancy_data.csv")

# Encode 'status'
le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])

# Define features and target
X = df.drop(columns=['life_expectancy', 'country'])
y = df['life_expectancy']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "life_expectancy_xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully!")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Print results
print(f" Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

import numpy as np

# Count how many predictions are within ±2 years
tolerance = 2
accuracy_within_2_years = np.mean(np.abs(y_pred - y_test) <= tolerance)

print(f"Custom Accuracy (within ±2 years): {accuracy_within_2_years * 100:.2f}%")


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load cleaned data and model
df = pd.read_csv("cleaned_life_expectancy_data.csv")
model = joblib.load("life_expectancy_xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Preserve original status and country for output
df_output = df.copy()

# Label encode 'status' ONLY for model input
df_model = df.copy()
le = LabelEncoder()
df_model['status'] = le.fit_transform(df_model['status'])  # 0 = Developed, 1 = Developing

# Separate features and target
X = df_model.drop(columns=['life_expectancy', 'country'])
y = df_model['life_expectancy']

# Scale features
X_scaled = scaler.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Get original rows that correspond to X_test
_, df_test_output = train_test_split(df_output, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test)

# Final dataframe for Power BI
print(type(df_test_output))
df_test_output = df_test_output.reset_index(drop=True)
df_test_output['Predicted_Life_Expectancy'] = y_pred
df_test_output['Error'] = df_test_output['life_expectancy'] - y_pred

# Save to CSV
df_test_output.to_csv("life_expectancy_powerbi_clean.csv", index=False)
print("✅ Power BI-ready CSV saved as 'life_expectancy_powerbi_clean.csv'")