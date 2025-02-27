import pandas as pd
import statsmodels.api as sm

# Load the data
data = pd.read_csv('modified_data.csv')

# Convert end date to datetime (using the format from your data)
data['End date'] = pd.to_datetime(data['End date'])

# Create time period indicator
def assign_time_period(date):
    if date < pd.Timestamp('2020-01-31'):
        return -1  # pre-covid
    elif date <= pd.Timestamp('2022-01-06'):
        return 0   # during-covid
    else:
        return 1   # post-covid

data['TimePeriod'] = data['End date'].apply(assign_time_period)

# Create default indicator based on supplier list charges
data['Default'] = (data['Charges have been made for remaining on supplier list'] == False).astype(int)

# Convert percentages to numeric values
dependent_vars = [
    '% Invoices paid within 30 days',
    '% Invoices paid between 31 and 60 days',
    '% Invoices paid later than 60 days',
    '% Invoices not paid within agreed terms'
]

# Add a constant for the regression models
X = sm.add_constant(data['TimePeriod'])

# First regression: Default vs Time Period
y_default = data['Default']
mask_default = ~(X.isna().any(axis=1) | y_default.isna())
X_clean_default = X[mask_default]
y_clean_default = y_default[mask_default]

model_default = sm.OLS(y_clean_default, X_clean_default).fit()
print("Regression results for Default vs Time Period:")
print(model_default.summary())
print("\n" + "="*50 + "\n")

# Run regression for each payment timing variable
for var in dependent_vars:
    y = data[var]
    # Remove any NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[mask]
    y_clean = y[mask]
    
    model = sm.OLS(y_clean, X_clean).fit()
    print(f"Regression results for {var}:")
    print(model.summary())
    print("\n" + "="*50 + "\n")

# Print summary statistics for each time period
print("Summary Statistics by Time Period:")
print("\nDefault Rates:")
print(data.groupby('TimePeriod')['Default'].mean())

print("\nPayment Timing Metrics:")
for var in dependent_vars:
    print(f"\n{var}:")
    print(data.groupby('TimePeriod')[var].mean())