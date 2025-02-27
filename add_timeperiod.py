import pandas as pd

# Load the dataset
data = pd.read_csv('modified_data.csv')

# Convert 'End date' to datetime
data['End date'] = pd.to_datetime(data['End date'])

# Define the function to categorize the time period
def categorize_time_period(end_date):
    if end_date < pd.Timestamp('2020-01-31'):
        return -1  # Pre-COVID
    elif end_date <= pd.Timestamp('2022-01-06'):
        return 0   # During-COVID
    else:
        return 1   # Post-COVID

# Apply the function to create the 'TimePeriod' column
data['TimePeriod'] = data['End date'].apply(categorize_time_period)

# Save the updated dataset to a new CSV file
data.to_csv('modified_data.csv', index=False)

print("TimePeriod column added and saved to 'modified_data.csv'.")