import pandas as pd
import statsmodels.api as sm

# Load the data
df = pd.read_csv('modified_data.csv')

# Display basic information and first few rows
print(df.info())
print(df.head())

# Convert 'End date' to datetime format
df["End date"] = pd.to_datetime(df["End date"], errors="coerce")

# Define time period categories
df["Time Period"] = df["End date"].apply(
    lambda x: -1 if x < pd.Timestamp("2020-01-31") else 
              0 if x <= pd.Timestamp("2022-01-06") else 1
)

# Check the distribution of the new variable
print(df["Time Period"].value_counts())

# Convert 'Charges have been made for remaining on supplier list' to binary Default variable
df["Default"] = df["Charges have been made for remaining on supplier list"].astype(bool).map({False: 1, True: 0})

# Check for missing values
print("Missing values in Default:", df["Default"].isna().sum())

# Verify the distribution of 'Default' variable
print(df["Default"].value_counts())

# Define dependent and independent variables
X = df[["Time Period"]]  # Independent variable
X = sm.add_constant(X)   # Add intercept
y = df["Default"]        # Dependent variable

# Run logistic regression
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Display summary
print(result.summary())
