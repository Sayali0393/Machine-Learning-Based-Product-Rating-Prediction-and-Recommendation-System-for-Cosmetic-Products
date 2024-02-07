import joblib
import pandas as pd

#Load the saved model
loaded_model = joblib.load('random_forest_regressor_model.joblib')

#Load new data
new_data_all = pd.read_csv('testing_data2.csv', low_memory=False)

#Drop unnecessary columns
column_to_drop = 'rating_pinfo'
new_data_all.drop(column_to_drop, axis=1, inplace=True)

#Make predictions
predictions = loaded_model.predict(new_data_all)

#Add predictions to the new data
new_data_all['Predictions'] = predictions

#Mapping for secondary categories
sec_cat_dict_map = {6: 'Moisturizers', 10: 'Treatments', 1: 'Eye Care', 3: 'Lip Balms & Treatments', 9: 'Sunscreen',
                    0: 'Cleansers', 11: 'Value & Gift Sets', 4: 'Masks', 5: 'Mini Size', 12: 'Wellness',
                    2: 'High Tech Tools', 7: 'Self Tanners', 8: 'Shop by Concern'}


#Replace encoded secondary category numbers with actual names
new_data_all['secondary_category_name'] = new_data_all['secondary_category_int'].map(sec_cat_dict_map)

#Get unique secondary categories
sec_cat_unique = new_data_all['secondary_category_int'].unique()

#Display top 5 recommended brands for each secondary category
for every_sec_cat in sec_cat_unique:
    sec_cat_name = sec_cat_dict_map.get(every_sec_cat, f'Unknown Category {every_sec_cat}')
    print(f"\nUsing Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: {sec_cat_name}")

    #Filter data for the current secondary category
    new_data = new_data_all[new_data_all['secondary_category_int'] == every_sec_cat]

    #Sort products by predicted scores for the current secondary category
    recommended_brands = new_data.groupby('brand_name2')['Predictions'].mean().sort_values(ascending=False).head(5)

    #Display top 5 recommended brands
    print(recommended_brands.reset_index()[['brand_name2', 'Predictions']])

#Save top 5 recommendations to a CSV file
top_5_recommendations_csv_filename = 'top_5_recommendations.csv'
new_data_all.to_csv(top_5_recommendations_csv_filename, index=False)
print(f'Top 5 recommended products saved to {top_5_recommendations_csv_filename}')

'''
"/Users/sayalidhobale/Documents/IT project/Pycharm/bin/Python" /Users/sayalidhobale/PycharmProjects/pythonProject2/Recommendationbasedonbrands_imp.py 

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Sunscreen
   brand_name2  Predictions
0         4567     4.600680
1         7044     4.301693
2         6792     4.241218
3         7405     3.907258
4         6902     3.396383

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: High Tech Tools
   brand_name2  Predictions
0         5413     4.551119
1         4717     4.494577
2         4045     4.442960
3         3268     4.346461
4         5794     4.280868

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Shop by Concern
   brand_name2  Predictions
0         5584     4.627803
1         6321     4.494357
2         3830     4.445440
3         5175     4.427450
4         6316     4.377150

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Mini Size
   brand_name2  Predictions
0         6535     4.524400
1         6156     4.380194
2         6089     4.351779
3         5806     4.347848
4         5537     4.308354

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Masks
   brand_name2  Predictions
0         4262     4.536924
1         5997     4.482685
2         6620     4.366072
3         6924     4.282253
4         6043     4.261958

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Treatments
   brand_name2  Predictions
0         3019     4.338193
1         7014     4.242205
2         7227     3.986537

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Cleansers
   brand_name2  Predictions
0         6968     4.472346
1         6753     4.442844
2         7179     4.255255
3         3281     4.059460
4         5805     4.031289

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Lip Balms & Treatments
   brand_name2  Predictions
0         4424     4.402309
1         7482     4.301753
2         7040     4.286183
3         6075     4.228730
4         7051     4.183616

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Moisturizers
   brand_name2  Predictions
0         6812     4.404409
1         7782     4.388599
2         4177     4.255757
3         5415     4.079894
4         4975     4.071158

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Self Tanners
   brand_name2  Predictions
0         5636     4.522099
1         7219     4.445787
2         4885     4.396770
3         6411     4.266085
4         5352     4.149003

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Eye Care
   brand_name2  Predictions
0         8017     4.466827
1         6139     4.402988
2         4606     4.371779
3         3777     4.357672
4         7062     4.356138

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Wellness
   brand_name2  Predictions
0         5270     4.313297
1         1979     4.273269
2         4052     4.261724
3         5566     4.097537

Using Random Forest, Recommending Top 5 Brands by Rating for Secondary Category: Value & Gift Sets
   brand_name2  Predictions
0         7601     4.490126
Top 5 recommended products saved to top_5_recommendations.csv

Process finished with exit code 0

'''
