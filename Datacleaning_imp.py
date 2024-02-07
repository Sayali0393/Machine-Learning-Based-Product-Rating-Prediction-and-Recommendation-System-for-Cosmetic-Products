#This file contins the tasks of cleaning the data and then visualizing it through heatmap, correlation matrix and scatterplots.
#importing the necessary libraries
import pandas as pd
import numpy as np
#getting all the files and combining data to obtain a dataframe
df = pd.read_csv('product_info.csv')
df1 = pd.read_csv('review1.csv', low_memory=False)
df2 = pd.read_csv('review2.csv', low_memory=False)
df3 = pd.read_csv('review3.csv', low_memory=False)
df4 = pd.read_csv('review4.csv', low_memory=False)
df5 = pd.read_csv('review5.csv', low_memory=False)

#checking if frames have similar data
set1_cols = set(list(df1.columns))
set2_cols = set(list(df4.columns))
difference = print(set1_cols - set2_cols)

frames= [df1, df2, df3, df4, df5]
result=pd.concat(frames)
result

#renaming columns for differentiating between columns obtained from different datasets
df = df.add_suffix('_pinfo')
df

#merging of dataframes using inner join
merged_df = pd.merge(df, result, how='inner', left_on='product_id_pinfo', right_on='product_id')
#getting information of the merged dataframe
merged_df.info()

m1=merged_df.copy(deep=True)
#converting the object datatype to string dataype
m1['brand_name_pinfo'] = m1['brand_name_pinfo'].astype("string")
m1['product_name_pinfo'] = m1['product_name_pinfo'].astype("string")
m1['product_id_pinfo'] = m1['product_id_pinfo'].astype("string")
m1['secondary_category_pinfo'] = m1['secondary_category_pinfo'].astype("string")
#extracting the float values from size column and copying it to new columns
pattern = r'(\d+\.\d+)'
m1[['size_oz_pinfo', 'size_ml_pinfo']] = m1['size_pinfo'].str.extractall(pattern).unstack()
m1[['size_oz_pinfo', 'size_ml_pinfo']] = m1[['size_oz_pinfo', 'size_ml_pinfo']].apply(pd.to_numeric, errors='coerce')

#checking for unique values in a column
print(f"primary_category_pinfo :  {m1['primary_category_pinfo'].unique()}")
print(f"skin_tone :  {m1['skin_tone'].unique()}")
print(f"eye_color :  {m1['eye_color'].unique()}")
print(f"skin_type :  {m1['skin_type'].unique()}")
print(f"hair_color :  {m1['hair_color'].unique()}")

#Copying columns and renaming them
m1['skin_tone_int'] = m1['skin_tone'].copy()
m1['eye_color_int'] = m1['eye_color'].copy()
m1['skin_type_int'] = m1['skin_type'].copy()
m1['hair_color_int'] = m1['hair_color'].copy()
m1['tertiary_category_pinfo_int'] = m1['tertiary_category_pinfo'].copy()

#Creating a copy of brand_id_pinfo and renaming it as brand_name2. Each brand name has it's own brand id. Instead of converting the brand names, it is easy to differentiate by brand id.
m1['brand_name2'] = m1['brand_id_pinfo'].copy()

#Encoding values in the newly formed columns
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
m1['skin_tone_int'] = label_encoder.fit_transform(m1['skin_tone_int'])
m1['eye_color_int'] = label_encoder.fit_transform(m1['eye_color_int'])
m1['skin_type_int'] = label_encoder.fit_transform(m1['skin_type_int'])
m1['hair_color_int'] = label_encoder.fit_transform(m1['hair_color_int'])
m1['tertiary_category_pinfo_int'] = label_encoder.fit_transform(m1['tertiary_category_pinfo_int'])
print(f"skin_tone_int :  {m1['skin_tone_int'].unique()}")

print(f"eye_color_int :  {m1['eye_color_int'].unique()}")
print(f"skin_type_int :  {m1['skin_type_int'].unique()}")
print(f"hair_color_int :  {m1['hair_color_int'].unique()}")
print(f"tertiary_category_pinfo_int :  {m1['tertiary_category_pinfo_int'].unique()}")

m1['secondary_category_int'] = m1['secondary_category_pinfo'].copy()
m1['secondary_category_int'] = label_encoder.fit_transform(m1['secondary_category_int'])
print(f"secondary_category_int :  {m1['secondary_category_int'].unique()}")

m1.info()

#Finding the missing data
df_missing = m1.isna().sum()
df_missing

#Finding the percentage of missing data
df_pct_missing = m1.isna().mean()
df_pct_missing

#searching for outliers
m1.kurt(numeric_only=True)

#checking for unnecessary data. First finding out the columns containing more than 50% of same values.
num_rows = len(m1)

for col in m1.columns:
    counts = m1[col].value_counts(dropna=False)
    top_pct = (counts / num_rows).iloc[0]

    if top_pct > 0.50:
        print('{0}:{1:2f}%'.format(col, top_pct * 100))
        print(counts)
        print()

#Searching for NaN values
print(m1.isna().sum())

#Replacing the NaN values in size_oz_pinfo column by the mean value
m1['size_oz_pinfo'] = m1['size_oz_pinfo'].fillna(m1['size_oz_pinfo'].mean())

#Replacing the NaN values in is_recommended column by 0.0 value. This column contains float datatype.
#1.0 value means that product is recommended whereas 0.0 means it isn't.
#Replacing NaN value by 0.0 will be mean that either the product is yet to be recommended.
m1['is_recommended'] = m1['is_recommended'].fillna(0.0)

m2=m1.copy(deep=True)
#Let's delete columns having high number of missing values. Considering columns which have more than 50% missing values.
#the colummns - variation_desc_pinfo, value_price_usd_pinfo, sale_price_usd_pinfo, size_ml_pinfo, helpfulness, child_max_price_pinfo, child_min_price_pinfo have high redundancy. These columns have all NaN values. Therefore, let's drop them.
#the columns - size_pinfo, product_id, product_name, brand_name, price_usd, product_name_pinfo, brand_name_pinfo, skin_tone, eye_color, skin_type, hair_color, tertiary_category_pinfo, secondary_category_pinfo, rating, brand_id_pinfo  are duplicate ones. New columns with desired datatypes have been added to the dataframe. Therefore, let's drop them.
#the columns - author_id, review_text, review_title, Unnamed: 0, submission_time, variation_type_pinfo, variation_value_pinfo, product_id_pinfo, highlights_pinfo don't contribute towards the task. Therefore, dropping them.
#the column - primary_category_pinfo has only 1 value after merging dataframes. Hence, it will be dropped.
#the columns - ingredients_pinfo, child_count_pinfo provide no insight to the task. Therefore, dropping it.
#lastly used columns_to_drop = ['brand_id_pinfo', 'total_pos_feedback_count', 'total_neg_feedback_count', 'skin_tone_int', 'eye_color_int', 'hair_color_int', 'sephora_exclusive_pinfo', 'out_of_stock_pinfo', 'online_only_pinfo', 'child_count_pinfo', 'rating', 'ingredients_pinfo', 'primary_category_pinfo', 'size_pinfo', 'variation_type_pinfo', 'submission_time', 'variation_value_pinfo', 'product_id_pinfo', 'tertiary_category_pinfo', 'highlights_pinfo', 'variation_desc_pinfo', 'value_price_usd_pinfo', 'sale_price_usd_pinfo', 'secondary_category_pinfo', 'size_ml_pinfo', 'product_id', 'product_name', 'brand_name', 'price_usd', 'product_name_pinfo', 'brand_name_pinfo', 'skin_tone', 'eye_color', 'skin_type', 'hair_color', 'helpfulness', 'child_max_price_pinfo', 'child_min_price_pinfo', 'author_id', 'review_text', 'review_title', 'Unnamed: 0']
#don't use columns_to_drop = ['skin_tone_int', 'eye_color_int', 'hair_color_int', 'brand_id_pinfo', 'child_count_pinfo', 'rating', 'sephora_exclusive_pinfo', 'is_recommended', 'online_only_pinfo', 'out_of_stock_pinfo', 'new_pinfo', 'limited_edition_pinfo', 'secondary_category_pinfo', 'tertiary_category_pinfo', 'size_pinfo', 'ingredients_pinfo', 'primary_category_pinfo', 'variation_type_pinfo', 'submission_time', 'variation_value_pinfo', 'product_id_pinfo', 'highlights_pinfo', 'variation_desc_pinfo', 'value_price_usd_pinfo', 'sale_price_usd_pinfo', 'size_ml_pinfo', 'product_id', 'product_name', 'brand_name', 'price_usd', 'product_name_pinfo', 'brand_name_pinfo', 'skin_tone', 'eye_color', 'skin_type', 'hair_color', 'helpfulness', 'child_max_price_pinfo', 'child_min_price_pinfo', 'author_id', 'review_text', 'review_title', 'Unnamed: 0']
columns_to_drop = ['brand_id_pinfo', 'child_count_pinfo', 'rating', 'ingredients_pinfo', 'primary_category_pinfo', 'size_pinfo', 'variation_type_pinfo', 'submission_time', 'variation_value_pinfo', 'product_id_pinfo', 'tertiary_category_pinfo', 'highlights_pinfo', 'variation_desc_pinfo', 'value_price_usd_pinfo', 'sale_price_usd_pinfo', 'secondary_category_pinfo', 'size_ml_pinfo', 'product_id', 'product_name', 'brand_name', 'price_usd', 'product_name_pinfo', 'brand_name_pinfo', 'skin_tone', 'eye_color', 'skin_type', 'hair_color', 'helpfulness', 'child_max_price_pinfo', 'child_min_price_pinfo', 'author_id', 'review_text', 'review_title', 'Unnamed: 0']
m2.drop(columns=columns_to_drop, inplace=True)
m2.info()

#the feedback columns have large number of outliers. These values depict their true nature. Hence, keeping them.

#Check for infinite values in the entire dataframe
is_infinite = np.isinf(m2)

#Check if there are any infinite values in the dataframe
if is_infinite.values.any():
    print("DataFrame contains infinite values.")
    # If you want to get the indices and columns where infinite values are present:
    rows, cols = np.where(is_infinite)
    for row, col in zip(rows, cols):
        print(f"Infinite value found at row {row}, column {m2.columns[col]}")

else:
    print("DataFrame does not contain infinite values.")

#Check for duplicate rows
duplicate_rows = m2[m2.duplicated()]

#Display the duplicate rows
print("\nDuplicate Rows:")
print(duplicate_rows)

#Drop duplicate rows
m2 = m2.drop_duplicates()

#Display the DataFrame after dropping duplicates
print("\nDataFrame after dropping duplicates:")
print(m2)

# Calculate the variance for each column in the DataFrame
#variances = m2.var()
#sorted_variances = variances.sort_values(ascending=False)
# Display the variances for each column
#print("Variances for each column:")
#print(sorted_variances)
m2.info()
m2.to_csv('m2_1.csv', index=False)

#show heatmap
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = m2.corr()
#print("Correlation Matrix:")
#print(corr_matrix)
plt.figure(figsize = (26,10))
sns.heatmap(corr_matrix, annot=True)
#plt.savefig("heatmap2.png")
plt.show()

#show correlation matrix
correlation_with_target = corr_matrix['rating_pinfo']
print("\nCorrelation with the Target Variable:")
print(correlation_with_target)

#Scatterplot
from scipy.optimize import curve_fit

num_features = 21

#Constant Y value (target variable)
constant_y = m2['rating_pinfo']

#Create a figure with subplots
fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, 3 * num_features))


def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c


#Generate scatterplots for each feature against the target variable
for i in range(num_features):
    #Extract the current feature
    feature_name = m2.columns[i]
    feature_values = m2[feature_name]

    #Create a scatterplot
    #axes[i].scatter(feature_values, constant_y, alpha=0.5, label=f'{feature_name} vs Target (sampled_data)')

    #Generate a sample of 50 random data points
    sampled_data = m2.sample(n=500, random_state=50)
    #axes[i].set_xlim(0, 5)

    #Create a scatterplot with the sampled data
    axes[i].scatter(sampled_data[feature_name], sampled_data['rating_pinfo'], label='Sampled Data Points', color='red')

    popt, pcov = curve_fit(quadratic_function, feature_values, constant_y)
    x_values = np.linspace(min(feature_values), max(feature_values), 100)
    axes[i].plot(x_values, quadratic_function(x_values, *popt), color='blue', label='Fitted Quadratic Curve')

    axes[i].set_title(f'{feature_name} vs Target')
    axes[i].set_xlabel(feature_name)
    axes[i].set_ylabel('rating_pinfo')
    axes[i].legend()

#Adjust layout and display the plots
plt.tight_layout()
plt.show()
