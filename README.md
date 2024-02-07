# Machine-Learning-Based-Product-Rating-Prediction-and-Recommendation-System-for-Cosmetic-Products

Cosmetics industry is one such industry that has seen consistent growth in the last decade[9]. It is expected to reach $599 billion USD in the year 2024. Modern technologies such as the Internet of Things, Cloud Computing, and AI/ML have significantly contributed to an overall growth of this industry. Considering these advancements, “Ratings” on cosmetic products are observed to influence the customer’s buying decisions. These ratings also help manufacturers understand their product performance and correspondingly update their business models. Various parameters such as the product quality, color, ingredients and trends affect a product’s rating. Developing a solution that considers such parameters as inputs and rating as an output will drive better decision making for both, customers and manufacturers.
For this study, Kaggle was chosen as the primary data resource. Data was acquired from Kaggle for this study (https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews). As per the source, Python script utilizing the Sephora and Bazaarvoice APIs was used for web scraping. BeautifulSoup and Pydantic libraries were applied to ensure data consistency. The datasets have 8000+ records of product information and more than 1 million records of user reviews. 

The first step was to read the records and convert them into dataframes. To differentiate among columns, a suffix of ‘_pinfo’ was added to column titles of product information dataframe. All the dataframes were then merged together for analyzing and visualization purposes using Pandas library. Info() method of Pandas proved useful in obtaining the summary concisely. The merged data had more than 1 million records or rows and 55 columns.
Next step undertaken was to convert non-numeric datatypes of columns to numeric for data analysis. Features such as ‘eye_color’ and ‘hair_color’ had some unique labels. To convert datatype of such features, LabelEncoder class of sklearn.preprocessing module was used and then added as new features. Also, wherever possible, astype() method was used to convert ‘object’ datatype to ‘string’. Float values of ‘oz’ and ‘ml’ from the ‘size_pinfo’ column were extracted and added to the dataframe as new features. 
The data was then checked for missing or NaN values, duplicate rows, outliers and infinite values. Features that contained more than 50% of duplicate values were identified. Columns were dropped mainly for three reasons -
Having more than 50% missing data or NaN values.
Providing no insight towards the task.
Old columns or features no longer serve due to the addition of new columns based on them.
Duplicate rows were removed as well.
The final dataframe contained 600,000+ records and 21 columns. 

To study the underlying diversity in data, trends and ranges, the data was visualized using three methods - heatmap, correlation matrix and scatterplots. The objective is to predict a product’s rating based on multiple variables using a robust machine learning based prediction model. And further provide recommendations to users of brands whose products have high ratings. The task revolves around the dependent variable - rating which has a continuous nature. Considering this essence, regression analysis is chosen to perform the task. Within the scope of the framework, rating serves as the response variable and price, brand name and the “loves” count serve as the predictor variables. Given the presence of numerous predictor variables and complex data relationships, the adoption of Random Forest Regression model was found to be appropriate to achieve the objective following a thorough research and testing.

Three scores were assessed to evaluate the model's performance: Their names and scores are -
Mean squared error (MSE) - 0.0077
Mean absolute error (MAE) - 0.0439
R squared (R²) - 0.9182

