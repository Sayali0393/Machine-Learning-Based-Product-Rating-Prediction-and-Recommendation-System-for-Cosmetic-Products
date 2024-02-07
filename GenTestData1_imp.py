import pandas as pd
import numpy as np

original_df = pd.read_csv('m2_1.csv', low_memory=False)

max_values = original_df.max()
min_values = original_df.min()

#Display max and min values for each column in the original DataFrame
for column in original_df.columns:
    max_value = original_df[column].max()
    min_value = original_df[column].min()
    print(f"Maximum value in '{column}': {max_value}")
    print(f"Minimum value in '{column}': {min_value}")
    print()

#Function to generate random data within the specified range with preserved data types
def generate_random_data_with_constraints(num_rows, max_values, min_values, original_df):
    data = {}
    for column in max_values.index:
        max_val = max_values[column]
        min_val = min_values[column]

        #Introduce more randomness by using a normal distribution
        mean_val = original_df[column].mean()
        std_val = original_df[column].std()

        #Generate random data and clip values to stay within the specified range
        data[column] = np.random.normal(loc=mean_val, scale=std_val, size=num_rows)
        data[column] = np.clip(data[column], min_val, max_val)

        #Preserve the data type from the original DataFrame
        data[column] = data[column].astype(original_df[column].dtype)

    return pd.DataFrame(data)

#Set the number of rows for the new DataFrame
num_rows_for_test_data = 5

#Generate random data with constraints for testing
testing_data = generate_random_data_with_constraints(num_rows_for_test_data, max_values, min_values, original_df)

#Display information about the testing data DataFrame
print("\nTesting Data Info:")
testing_data.info()

#Save testing data to CSV
testing_data.to_csv('testing_data2.csv', index=False)


