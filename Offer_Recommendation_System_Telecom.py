# %% [markdown]
# # Package requirements & imports

# %%
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import *


# %% [markdown]
# # Data import

# %%
csv_file_path = './data/Telecom_data.csv'
df = pd.read_csv(csv_file_path)
df.head(10)

# %% [markdown]
# # **Exploratory Data Analysis**

# %% [markdown]
# ## **Data Exploration**
# 
# Data exploration is a critical step in the data analysis process. Data exploration is important because it helps to provide a solid foundation for subsequent data analysis tasks, hypothesis testing and data visualization.
# 
# Data exploration is also important because it can help you to identify an appropriate approach for analyzing the data.
# 
# Here are the various functions that help us explore and understand the data.
# 
# * Shape: Shape is used to identify the dimensions of the dataset. It gives the number of rows and columns present in the dataset. Knowing the dimensions of the dataset is important to understand the amount of data available for analysis and to determine the feasibility of different methods of analysis.
# 
# * Head: The head function is used to display the top five rows of the dataset. It helps us to understand the structure and organization of the dataset. This function gives an idea of what data is present in the dataset, what the column headers are, and how the data is organized.
# 
# * Tail: The tail function is used to display the bottom five rows of the dataset. It provides the same information as the head function but for the bottom rows. The tail function is particularly useful when dealing with large datasets, as it can be time-consuming to scroll through all the rows.
# 
# * Describe: The describe function provides a summary of the numerical columns in the dataset. It includes the count, mean, standard deviation, minimum, and maximum values, as well as the quartiles. It helps to understand the distribution of the data, the presence of any outliers, and potential issues that can affect the model's accuracy.
# 
# * Isnull: The isnull function is used to identify missing values in the dataset. It returns a Boolean value for each cell, indicating whether it is null or not. This function is useful to identify the presence of missing data, which can be problematic for regression analysis.
# 
# * Dropna: The dropna function is used to remove rows or columns with missing data. It is used to remove any observations or variables with missing data, which can lead to biased results in the regression analysis. The dropna function is used after identifying the missing data with the isnull function.
# 
# * Columns: The .columns method is a built-in function that is used to display the column names of a pandas DataFrame or Series. It returns an array-like object that contains the names of the columns in the order in which they appear in the original DataFrame or Series. It can be used to obtain a quick overview of the variables in a dataset and their names.

# %%
df.shape

# %%
df.columns

# %% [markdown]
# ## **Data Dictionary**
# 
# 
# 
# | Column name	 | Description|
# | ----- | ----- |
# | Customer ID	 | Unique identifier for each customer |
# | Month | Calendar Month- 1:12 | 
# | Month of Joining |	Calender Month -1:14, Month for which the data is captured|
# | zip_code |	Zip Code|
# |Gender |	Gender|
# | Age |	Age(Years)|
# | Married |	Marital Status |
# |Dependents | Dependents - Binary |
# | Number of Dependents |	Number of Dependents|
# |Location ID |	Location ID|
# |Service ID	 |Service ID|
# |state|	State|
# |county	|County|
# |timezone	|Timezone|
# |area_codes|	Area Code|
# |country	|Country|
# |latitude|	Latitude|
# |longitude	|Longitude|
# |arpu|	Average revenue per user|
# |roam_ic	|Roaming incoming calls in minutes|
# |roam_og	|Roaming outgoing calls in minutes|
# |loc_og_t2t|	Local outgoing calls within same network in minutes|
# |loc_og_t2m	|Local outgoing calls outside network in minutes(outside same + partner network)|
# |loc_og_t2f|	Local outgoing calls with Partner network in minutes|
# |loc_og_t2c	|Local outgoing calls with Call Center in minutes|
# |std_og_t2t|	STD outgoing calls within same network in minutes|
# |std_og_t2m|	STD outgoing calls outside network in minutes(outside same + partner network)|
# |std_og_t2f|	STD outgoing calls with Partner network in minutes|
# |std_og_t2c	|STD outgoing calls with Call Center in minutes|
# |isd_og|	ISD Outgoing calls|
# |spl_og	|Special Outgoing calls|
# |og_others|	Other Outgoing Calls|
# |loc_ic_t2t|	Local incoming calls within same network in minutes|
# |loc_ic_t2m|	Local incoming calls outside network in minutes(outside same + partner network)|
# |loc_ic_t2f	|Local incoming calls with Partner network in minutes|
# |std_ic_t2t	|STD incoming calls within same network in minutes|
# |std_ic_t2m	|STD incoming calls outside network in minutes(outside same + partner network)|
# |std_ic_t2f|	STD incoming calls with Partner network in minutes|
# |std_ic_t2o|	STD incoming calls operators other networks in minutes|
# |spl_ic|	Special Incoming calls in minutes|
# |isd_ic|	ISD Incoming calls in minutes|
# |ic_others|	Other Incoming Calls|
# |total_rech_amt|	Total Recharge Amount in Local Currency|
# |total_rech_data|	Total Recharge Amount for Data in Local Currency
# |vol_4g|	4G Internet Used in GB|
# |vol_5g|	5G Internet used in GB|
# |arpu_5g|	Average revenue per user over 5G network|
# |arpu_4g|	Average revenue per user over 4G network|
# |night_pck_user|	Is Night Pack User(Specific Scheme)|
# |fb_user|	Social Networking scheme|
# |aug_vbc_5g|	Volume Based cost for 5G network (outside the scheme paid based on extra usage)|
# |offer|	Offer Given to User|
# |Referred a Friend|	Referred a Friend : Binary|
# |Number of Referrals|	Number of Referrals|
# |Phone Service|	Phone Service: Binary|
# |Multiple Lines|	Multiple Lines for phone service: Binary|
# |Internet Service|	Internet Service: Binary|
# |Internet Type|	Internet Type|
# |Streaming Data Consumption|	Streaming Data Consumption|
# |Online Security|	Online Security|
# |Online Backup|	Online Backup|
# |Device Protection Plan|	Device Protection Plan|
# |Premium Tech Support|	Premium Tech Support|
# |Streaming TV|	Streaming TV|
# |Streaming Movies|	Streaming Movies|
# |Streaming Music|	Streaming Music|
# |Unlimited Data|	Unlimited Data|
# |Payment Method|	Payment Method|
# |Status ID|	Status ID|
# |Satisfaction Score|	Satisfaction Score|
# |Churn Category|	Churn Category|
# |Churn Reason|	Churn Reason|
# |Customer Status|	Customer Status|
# |Churn Value|	Binary Churn Value
# 
# 

# %%
df.info()

# %%
df["arpu_4g"].unique()

# %%
df["arpu_4g"].unique()

# %%
df['offer'].unique()

# %%
# Taking a look at the offer distribution
dfg = df.groupby('offer').agg({'Customer ID':'count'}).reset_index()
dfg['% Total'] = dfg['Customer ID']/dfg['Customer ID'].sum() #this creates a % of total column
dfg['% Total'] = dfg['% Total'].apply(lambda x: '{:.2%}'.format(x)) #this function simply formats the column to %
dfg #this displays the dataframe

# %% [markdown]
# **Observation**
# * The Offers seems to be evenly distributed amongst customers
# * There are about 76% users who did not receive any offer from the company
# 

# %% [markdown]
# **Questions**
# 
# * Is there any way to check impact of offers on churn? 
# * How many customer churned as they were not given any offer?

# %%
dfg2 = df.groupby(['offer','Customer Status']).agg({'Customer ID':'count'}).reset_index()
pivoted_dfg2 = dfg2.pivot(index='offer', columns='Customer Status', values='Customer ID')
pivoted_dfg2 = pivoted_dfg2.reset_index()
pivoted_dfg2['Churn Rate'] = pivoted_dfg2['Churned']/(pivoted_dfg2['Churned'] + pivoted_dfg2['Stayed'])
pivoted_dfg2['Churn Rate'] = pivoted_dfg2['Churn Rate'].apply(lambda x: '{:.2%}'.format(x)) #this function simply formats the column to %
pivoted_dfg2

# %% [markdown]
# **Observations**
# 
# * churn rate seems to be similar amongst customers regardless of the offer they received -> this tells us that maybe offers are not being tailored enough to groups
# 

# %%
# Taking a look at the churn category
dfg2 = df.groupby(['Churn Category',]).agg({'Customer ID':'count'}).reset_index()
dfg2['% Total'] = dfg2['Customer ID']/dfg2['Customer ID'].sum() #this creates a % of total column
dfg2['% Total'] = dfg2['% Total'].apply(lambda x: '{:.2%}'.format(x)) #this function simply formats the column to %
dfg2 #this displays the dataframe

# %% [markdown]
# **Observations**
# 
# * The Churn Category for Competitor, Dissatisfaction, Price, Support have higher customers
# * We can give them specific offers which may lead them to stay rather than churning
# 

# %% [markdown]
# ****Questions****
# 
# * Is there any better way to recommend offers to customers which can impact less churn rate in future?

# %% [markdown]
# # **Data Processing**

# %% [markdown]
# ## **Missing Value Detection and Imputation**

# %% [markdown]
# We previously saw there are some missing values in the data. Lets have a look into that now.

# %%
# Creating a missing value df with the null values of our original dataframe
percent_missing = df.isna().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing.values})

#sorting the dataframe by percent missing value 
missing_value_df.sort_values(by='percent_missing',ascending=False)

# %% [markdown]
# **Observation**
# 
# 
# 
# *  Columns 'fb_user' and 'night_pck_user' have more 50% missing value. We will simply drop this from our dataframe
# * According to data dictionary 'Internet Type' and 'total_rech_data' seems to correlated.
# *   We need to check for columns 'Internet Type' and 'total_rech_data' and impute missing values if possible
# 
# 
# 

# %%
#dropping the variables with more than 50% null values
df=df.drop(columns=['fb_user','night_pck_user'])

# %%
# Null values in total recharge data
df['total_rech_data'].isna().sum()

# %%
# Null values in Internet Type
df['Internet Type'].isna().sum()

# %% [markdown]
# **Observation:**
# 
# *  These missing values may represent customers who have not recharged their account or have recharged but the information has not been recorded.
# 
# * It is possible that customers with missing recharge data are those who received free data service, and therefore did not need to recharge their account. Alternatively, it is possible that the missing values are due to technical issues, such as data recording errors or system failures.

# %%
# Checking the value counts of Internet Service where total recharge data was null
df[df['total_rech_data'].isna()]['Internet Service'].value_counts(dropna=False)

# %% [markdown]
# **Observation**:
# 
# * It turns out that all customers with missing recharge data have opted for internet service, the next step could be to check if they have used it or not.

# %%
# Let's check unlimited data column
df[(df['total_rech_data'].isna())]['Unlimited Data'].value_counts()

# %%
# Lets check Average Revenue for 4g and 5g when there is no recharge for data
df[(df['total_rech_data'].isna())][['arpu_4g','arpu_5g']].value_counts()

# %% [markdown]
# **Observation**:
# 
# * We can fill the missing values in the total_rech_data column with 0 when the arpu (Average Revenue Per User) is not applicable. This is because the arpu is a measure of the revenue generated per user, and if it is not applicable, it may indicate that the user is not generating any revenue for the company. In such cases, it is reasonable to assume that the total data recharge amount is 0
# * It is advisable to check with the business before making this decision 

# %%
# Replacing all values of total recharge data= 0 where arpu 4g and 5g are not applicable
df.loc[(df['arpu_4g']=='Not Applicable') | (df['arpu_5g']=='Not Applicable'),'total_rech_data']=0

# %%
# Missing value percentage after imputation
df['total_rech_data'].isna().sum()/df.shape[0]

# %% [markdown]
# We cannot fill other values with 0 because they have some ARPU to consider.

# %%
# Calculate the mean of 'total_rech_data' where either 'arpu_4g' or 'arpu_5g' is not equal to 'Not Applicable'
arpu_data_mean=df.loc[(df['arpu_4g']!='Not Applicable') | (df['arpu_5g']!='Not Applicable'),'total_rech_data'].mean()
arpu_data_mean

# %%
# Fill NaN values in 'total_rech_data' with the mean of 'total_rech_data' where either 'arpu_4g' or 'arpu_5g' is not equal to 'Not Applicable'
df['total_rech_data']=df['total_rech_data'].fillna(arpu_data_mean)

# %%
df['total_rech_data'].isna().sum()

# %% [markdown]
# There are no more missing values in the column ''total_rech_data'

# %%
# Check the value counts for Internet Type
df['Internet Type'].value_counts(dropna=False)

# %%
# Check value counts for Internet Service where Internet Type is null
df[df['Internet Type'].isna()]['Internet Service'].value_counts(dropna=False)

# %% [markdown]
# All null values in Internet Type does not have Internet Service. Let's fill these null values with Not Applicable.

# %%
# Filling Null values in Internet Type 
df['Internet Type']=df['Internet Type'].fillna('Not Applicable')

# %% [markdown]
# Replace 'Not Applicable' with 0 in both 'arpu_4g' and 'arpu_5g and convert them to float

# %%
# Replace 'Not Applicable' with 0 in 'arpu_4g'
df['arpu_4g'] = df['arpu_4g'].replace('Not Applicable', 0)

# Replace 'Not Applicable' with 0 in 'arpu_5g'
df['arpu_5g'] = df['arpu_5g'].replace('Not Applicable', 0)

# Convert 'arpu_4g' to float data type
df['arpu_4g'] = df['arpu_4g'].astype(float)

# Convert 'arpu_5g' to float data type
df['arpu_5g'] = df['arpu_5g'].astype(float)

# %% [markdown]
# ## **Outlier Detection and Imputation**
# 
# 
# Outlier detection is a critical data analysis technique that involves identifying and removing data points that are significantly different from the rest of the data. Outliers are data points that lie far away from the rest of the data, and they can significantly influence the statistical analysis and machine learning models' performance. Therefore, identifying and removing outliers is essential to ensure accurate and reliable data analysis results.
# 
# There are two main approaches for outlier detection: parametric and non-parametric.
# 
# * Parametric Methods:
# Parametric methods assume that the data follows a specific distribution, such as a normal distribution. In this approach, outliers are identified by calculating the distance of each data point from the mean of the distribution in terms of the number of standard deviations. Data points that are beyond a certain number of standard deviations (usually three or more) are considered as outliers.
# 
# One common parametric method is the Z-score method, which calculates the distance of each data point from the mean in terms of standard deviations.
# Parametric methods can be useful when the data follows a known distribution, but they may not be effective when the data is not normally distributed.
# 
# * Non-Parametric Methods:
# Non-parametric methods do not assume any specific distribution of the data. Instead, they rely on the rank or order of the data points. In this approach, outliers are identified by comparing the values of each data point with the values of other data points. Data points that are significantly different from other data points are considered as outliers.
# 
# Quantiles are an important concept in non-parametric outlier detection methods. They represent values that divide a dataset into equal-sized parts, such as quarters or thirds. The most commonly used quantiles are the median (which divides the data into two equal parts), the first quartile (which divides the data into the lowest 25% and the highest 75%), and the third quartile (which divides the data into the lowest 75% and the highest 25%).
# 
# The interquartile range (IQR) is another important concept related to quantiles. It is defined as the difference between the third and first quartiles and represents the middle 50% of the data. The IQR can be used to identify outliers by defining a range (known as the Tukey's fence) beyond which any data points are considered outliers.
# Non-parametric methods can be useful when the data is not normally distributed or when the distribution is unknown.

# %%
# List of continuous columns
cts_cols=['Age','Number of Dependents',
       'roam_ic', 'roam_og', 'loc_og_t2t',
       'loc_og_t2m', 'loc_og_t2f', 'loc_og_t2c', 'std_og_t2t', 'std_og_t2m',
       'std_og_t2f', 'std_og_t2c', 'isd_og', 'spl_og', 'og_others',
       'loc_ic_t2t', 'loc_ic_t2m', 'loc_ic_t2f', 'std_ic_t2t', 'std_ic_t2m',
       'std_ic_t2f', 'std_ic_t2o', 'spl_ic', 'isd_ic', 'ic_others',
       'total_rech_amt', 'total_rech_data', 'vol_4g', 'vol_5g', 'arpu_5g',
       'arpu_4g', 'arpu', 'aug_vbc_5g', 'Number of Referrals','Satisfaction Score',
       'Streaming Data Consumption']   


# %%
# Create an empty dataframe with columns as cts_cols and index as quantiles
quantile_df=pd.DataFrame(columns=cts_cols,index=[0.1,0.25,0.5,0.75,0.8,0.9,0.95,0.97,0.99])

# for each column in cts_cols, calculate the corresponding quantiles and store them in the quantile_df
for col in cts_cols:
   quantile_df[col]=df[col].quantile([0.1,0.25,0.5,0.75,0.8,0.9,0.95,0.97,0.99])

# %% [markdown]
# By calculating quantiles for each continuous variable in the dataset, we are trying to get an idea about the spread and distribution of the data. Specifically, we are interested in identifying potential outliers in the data.
# 
# Quantiles divide a distribution into equal proportions. For instance, the 0.25 quantile is the value below which 25% of the observations fall and the 0.75 quantile is the value below which 75% of the observations fall. By calculating quantiles at various levels, we can get a better understanding of the distribution of the data and identify any observations that are too far away from the rest of the data.
# 
# These quantiles can be used as thresholds to identify potential outliers in the data. Observations with values beyond these thresholds can be considered as potential outliers and further investigation can be carried out to determine if they are true outliers or not.

# %%
# Let's check out the quantiles df
quantile_df

# %% [markdown]
# **Observation**
# 
# The variables vol_5g, arpu_4g, and arpu_5g seems to have some abrupt values

# %%
# Checking further
df['arpu_4g'].quantile([0.75,0.8,0.9,0.95,0.97,0.99,0.999])

# %%
# Calculate the proportion of rows in the DataFrame where the value in the 'arpu_4g' column is equal to 254687
df[df['arpu_4g']==254687].shape[0]/df.shape[0]

# %% [markdown]
# Let's see what is the value of 'total_rech_data' for these observations.

# %%
# Get the value counts of 'total_rech_data' for observations where the value in the 'arpu_4g' column is equal to 254687
df[df['arpu_4g']==254687]['total_rech_data'].value_counts()

# %% [markdown]
# Now, since the recharge amount is 0 and there is no ARPU, let's replace it with 0.

# %%
# Replace the outlier value 254687 in the 'arpu_4g' column of the dataframe 'df' with 0.
df['arpu_4g']=df['arpu_4g'].replace(254687,0)

# %%
# Checking further
df['arpu_4g'].quantile([0.75,0.8,0.9,0.95,0.97,0.99,0.999])

# %%
# Filter by 'arpu_4g' value of 87978 and count unique values in 'total_rech_data' column
df[df['arpu_4g']==87978]['total_rech_data'].value_counts()

# %% [markdown]
# All rows in the dataframe with an 'arpu_4g' value of 87978 have 0 value in the 'total_rech_data' column, indicating that these are likely outliers. Therefore, we have decided to replace the 'arpu_4g' value for these rows with 0.

# %%
# Replace the values with 0
df['arpu_4g']=df['arpu_4g'].replace(87978,0)

# %%
# Checking the quantiles again
df['arpu_4g'].quantile([0.75,0.8,0.9,0.95,0.97,0.99,0.999])

# %% [markdown]
# This seems to be fairly good now

# %%
# Get the value counts of 'total_rech_data' for observations where the value in the 'arpu_5g' column is equal to 254687
df[df['arpu_5g']==254687]['total_rech_data'].value_counts()

# %%
# Get the value counts of 'total_rech_data' for observations where the value in the 'arpu_5g' column is equal to 87978
df[df['arpu_5g']==87978]['total_rech_data'].value_counts()

# %%
# Replacing the values with 0 where total recharge data is 0
df['arpu_5g']=df['arpu_5g'].replace([87978,254687],0)

# %%
# Check the quantiles of ARPU 5G
df['arpu_5g'].quantile([0.75,0.8,0.9,0.95,0.97,0.99,0.999])

# %% [markdown]
# This seems to be fairly good now

# %%
# Check the quantiles of Volume of 5G data
df['vol_5g'].quantile([0.75,0.8,0.9,0.95,0.97,0.98,0.99,0.999])

# %%
# Lets see the recharge data value for vol_5g more than 87978
df[df['vol_5g']>=87978]['total_rech_data'].value_counts()

# %%
# Proportion of these values
df[df['vol_5g']>=87978]['total_rech_data'].value_counts()/df.shape[0]

# %% [markdown]
# **Observation**:
# 
# There is a presence of 2% outliers in vol 5g, where the values are very high, but their total recharge data is 0. We will fill these outliers with 0, and below are some possible reasons why this could be:
# 
# * Data recording error: It is possible that there was an error in recording the recharge data for these outliers, leading to an incorrect value of 0. In this case, it would make sense to fill the outliers with 0, as this is likely the correct value.
# 
# * Promotions or bonuses: Another possibility is that these customers received promotions or bonuses that allowed them to use the service without recharging, leading to a total recharge data of 0. However, these customers may still be using the service heavily, leading to the high values in vol 5g. In this case, filling the outliers with 0 would make sense as it accurately reflects the lack of recharge data.

# %%
# Replace the outlier values
df['vol_5g']=df['vol_5g'].replace([87978,254687],0)

# %%
# Check the quantiles of Volume of 5G data
df['vol_5g'].quantile([0.75,0.8,0.9,0.95,0.97,0.98,0.99,0.999])

# %% [markdown]
# It seems good now

# %% [markdown]
# Lets store this processed data for further use.

# %%
df.to_csv('./data/output/processed_telecom_offer_data.csv',index=False)

# %% [markdown]
# # **Feature engineering**

# %% [markdown]
# Let's start by doing some variable selection and transformation. For selection, at this stage, we are going to use some business judgement to stick to what is possible to work with all the variables.

# %% [markdown]
# **1. Splitting the dataset into a training and production dataset:**
# 
# - Training: part of the customers who received offers which will be used to train the model
# - Production: customers who did not received offers to whom we'd like to then offer something

# %%
# Let's split our dataframe in a training and production dataset:
def split_dataframe(data):
    train = data[data['offer']!='No Offer']
    production = data[data['offer']=='No Offer']
    return train, production

# %%
train, production = split_dataframe(df)

# %%
train.shape,production.shape

# %% [markdown]
# **Questions**
# 
# 
# Why we split the dataframe into such unusual technique?
# 
# Here we are not dealing with traditional train test split method as we are building an unsupervised collaborative recommender system.
# Now to make model learn we need to pass all the data with respect to offers.

# %% [markdown]
# we are creating 2 dataframes for each train and production, that have the Customer ID as a join key. This will help us manipulate features, and also trace them back to a particular customer;
# 

# %%
#This help us identify the customer and the business outcomes
id_variables = ['Customer ID', 'Month','Month of Joining','offer','Churn Category',
       'Churn Reason', 'Customer Status', 'Churn Value']


#This helps us identify the different profiles of customers
selected_variables = ['Customer ID', 'Month', 'Month of Joining', 'Gender', 'Age',
                      'Married', 'Number of Dependents', 'area_codes','roam_ic', 'roam_og',
                      'loc_og_t2t','loc_og_t2m', 'loc_og_t2f', 'loc_og_t2c', 'std_og_t2t', 'std_og_t2m',
                      'std_og_t2f', 'std_og_t2c', 'isd_og', 'spl_og', 'og_others',
                      'loc_ic_t2t', 'loc_ic_t2m', 'loc_ic_t2f', 'std_ic_t2t', 'std_ic_t2m',
                      'std_ic_t2f', 'std_ic_t2o', 'spl_ic', 'isd_ic', 'ic_others',
                      'total_rech_amt', 'total_rech_data', 'vol_4g', 'vol_5g', 'arpu_5g',
                      'arpu_4g', 'arpu', 'aug_vbc_5g','Number of Referrals', 'Phone Service',
                      'Multiple Lines', 'Internet Service', 'Internet Type',
                      'Streaming Data Consumption', 'Online Security', 'Online Backup',
                      'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
                      'Streaming Movies', 'Streaming Music', 'Unlimited Data',
                      'Payment Method']

train_id=train[id_variables]
train_feat=train[selected_variables]

prod_id=production[id_variables]
prod_feat=production[selected_variables]

# %% [markdown]
# In the code above, what we have essentially eliminated are complex variables like latitude, longitude and timezone because they could be represented by the area_codes variable, that represents location.

# %% [markdown]
# 
# **2. Converting the Month of Joining into a customer tenure**

# %%
train_feat['tenure'] = train_feat['Month']- train_feat['Month of Joining']
train_feat['tenure'].describe()

# %%
prod_feat['tenure'] = prod_feat['Month']- prod_feat['Month of Joining']
prod_feat['tenure'].describe()

# %% [markdown]
# **3.Transforming Categorical Variables**
# 
# Transforming variables is an important step in the data preprocessing pipeline of machine learning, as it helps to convert the data into a format that is suitable for analysis and modeling. There are several ways to transform variables, depending on the type and nature of the data.
# 
# Categorical variables, for example, are variables that take on discrete values from a finite set of categories, such as colors, gender, or occupation. One common way to transform categorical variables is through one-hot encoding. One-hot encoding involves creating a new binary variable for each category in the original variable, where the value is 1 if the observation belongs to that category and 0 otherwise. This approach is useful when the categories have no natural order or ranking.
# 
# Another way to transform categorical variables is through label encoding. Label encoding involves assigning a unique integer value to each category in the variable. This approach is useful when the categories have a natural order or ranking, such as low, medium, and high.
# 
# 

# %%
# Now we need to transform the features of the feature store.
def encode_categorical_features(train_df,prod_df):
    # Get a list of all categorical columns
    cat_columns = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Encode each categorical column
    for col in cat_columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        prod_df[col]= le.transform(prod_df[col])
    return train_df, prod_df

# %%
#excluding the customer ID so it doesn't get encoded
train_label_data=train_feat[train_feat.columns.difference(['Customer ID','Month','Month of Joining'])]
prod_label_data=prod_feat[prod_feat.columns.difference(['Customer ID','Month','Month of Joining'])]
train_feat_enc, prod_feat_enc = encode_categorical_features(train_label_data,prod_label_data)

# %%
##bringing back the customer ids keys
train_feat_enc['Customer ID'] = train_feat['Customer ID'] #bringing back the customer id
train_feat_enc['Month'] = train_feat['Month'] #bringing back the Month
train_feat_enc['Month of Joining'] = train_feat['Month of Joining'] #bringing back the Month of joining

prod_feat_enc['Customer ID'] = prod_feat['Customer ID'] #bringing back the customer id
prod_feat_enc['Month'] = prod_feat['Month'] #bringing back the Month
prod_feat_enc['Month of Joining'] = prod_feat['Month of Joining'] #bringing back the Month of joining


# %% [markdown]
# Taking a look at the final list of variables
# 

# %%
train_feat_enc.describe().transpose()

# %%
train = pd.merge(train_feat_enc,train_id[['Customer ID','Month','Month of Joining','Churn Value','offer']],how = 'inner',on=['Customer ID','Month','Month of Joining'])
production = pd.merge(prod_feat_enc,prod_id[['Customer ID','Month','Month of Joining','Churn Value','offer']],how = 'inner',on=['Customer ID','Month','Month of Joining'])


# %%
## This help us check if we did not duplicate anything in the columns

print(len(train))
print(len(production))

# %% [markdown]
# # **Model Building and Testing**

# %% [markdown]
# ## **Collaborative filtering**
# 
# Collaborative filtering is a type of recommendation system that uses user feedback to make personalized recommendations for items. It works by finding similarities between users or items and using those similarities to make predictions about what a user might like or dislike.
# 
# Collaborative filtering can be broken down into two main types: user-based and item-based. In user-based collaborative filtering, similarities are calculated between users based on their past interactions with items. In item-based collaborative filtering, similarities are calculated between items based on how often they are interacted with by the same users.
# 
# Distance measures are commonly used in collaborative filtering to calculate similarities between users or items. Some common distance measures used in collaborative filtering include Manhattan distance, Euclidean distance, and cosine similarity.
# 
# The basic idea behind using distance measures in collaborative filtering is that similar users or items will be close together in the feature space defined by the data. For example, if we are recommending movies to users based on their past movie ratings, we might represent each user as a vector of their ratings, with each rating corresponding to a different movie. We could then calculate the distance between two users' rating vectors using a distance measure like cosine similarity or Euclidean distance. Users who have similar ratings for the same movies will be closer together in this feature space and therefore will have a smaller distance between them.
# 
# Once we have calculated similarities between users or items, we can use those similarities to make predictions about what a user might like or dislike. For example, if we have calculated the similarity between two users and we know that one of them likes a certain movie, we can predict that the other user might also like that movie based on their similarity.
# 
# There are many variations of collaborative filtering that use different distance measures and algorithms for finding similarities between users or items. Some examples include k-nearest neighbors (k-NN), which uses the distances between users to find the k users who are most similar to a given user, and matrix factorization, which uses linear algebra techniques to decompose the user-item interaction matrix into lower-dimensional matrices that capture user and item characteristics.
# 
# In summary, collaborative filtering is a type of recommendation system that uses user feedback to make personalized recommendations for items. It uses distance measures to calculate similarities between users or items, which are then used to make predictions about what a user might like or dislike.

# %% [markdown]
# ## **Mathematical explanation for distance measure**
# 
# Manhattan, cosine, and Euclidean distance are different distance metrics used in machine learning and data science.
# 
# ### **Manhattan distance:**
# Manhattan distance, also known as taxicab distance or L1 distance, is a measure of the distance between two points in a n-dimensional space. It is called Manhattan distance because it is analogous to the distance a taxi would travel on the streets of Manhattan, where you can only move in straight lines along the grid.
# The formula for Manhattan distance between two points P and Q in n-dimensional space is:
# 
# \begin{equation}
# d(P,Q) = |x_1-y_1| + |x_2-y_2| + ... + |x_n-y_n|
# \end{equation}
# 
# where $x_1, x_2, ..., x_n$ are the coordinates of point $P$ and $y_1, y_2, ..., y_n$ are the coordinates of point $Q$.
# 

# %% [markdown]
# ### **Cosine similarity:**
# Cosine similarity is a measure of the similarity between two non-zero vectors of an inner product space. It is the cosine of the angle between the two vectors and ranges from -1 to 1. A value of 1 indicates that the two vectors are identical, while a value of -1 indicates that they are completely dissimilar.
# The formula for cosine similarity between two vectors A and B is:
# 
# \begin{equation}
# \text{cosine   similarity}(A, B) = \frac{A * B }{ ||A|| * ||B||}
# \end{equation}
# 
# 
# 
# where $A * B$ is the dot product of vectors A and B, and $||A||$ and $||B||$ are the magnitudes of vectors A and B, respectively.

# %%
def get_recommended_offers (df:pd.DataFrame, df_id:pd.DataFrame,customer_id:str,month:int,distance_func:str,n,minimal_threshold:float,max_offers_to_return:int):
    """
    This function takes as parameters:
    1. the dataframe where we'll be getting our data
    2. the customer identifiers Customer Id and the Month we want to make an offer for
    3. the distance function we want to use to calculate similaries between customers (see explanation below on how to chose it)
    4. The number of other customers we want to base our recommendations on
    5. The minimal threshold of prevalence of a given offer, in the similar group of customers, for it to be considered for recommendation (see explanation below on how to chose it)
    
    It returns:
    An array with the list of offers that we could recommend to this customer
    """

    # extract the feature vectors of all customers
    features = list(df.columns.difference(['Customer ID','Month','Month of Joining','offer']))
    X = df[features].values

    # extract the feature vector of the given customer
    index = df[(df['Customer ID'] == customer_id) & (df['Month']==month)].index[0]
    x = X[index]

    # compute the distances between the feature vectors
    if distance_func == 'euclidean':
      distances = euclidean_distances(X, x.reshape(1, -1)).flatten()
    elif distance_func == 'manhattan':
      distances = manhattan_distances(X, x.reshape(1, -1)).flatten()
    elif distance_func == 'cosine':
      distances = 1 - cosine_similarity(X, x.reshape(1, -1)).flatten()
    else:
      raise ValueError('Invalid distance function specified.')

    # find the indices of the n customers with lowest distance
    most_similar_indices = distances.argsort()[:n]
            
    # extract the customer data for the most similar customers
    similar_customers = df.iloc[most_similar_indices]

    # merge with the id dataframe to select only the customers who did not churn
    similar_customers = pd.merge(similar_customers,df_id[['Customer ID','Month of Joining','Month','Churn Value']],on=['Customer ID','Month of Joining','Month','Churn Value'])

    # select the customers that did not churn
    similar_customers = similar_customers[similar_customers['Churn Value']==0]

    #count the top offers of the non-churned customers
    top_offers = similar_customers[['Customer ID','offer']].groupby(['offer']).agg({'Customer ID':'count'}).reset_index().sort_values(by = 'Customer ID', ascending = False)
    top_offers['perc_total'] = top_offers['Customer ID']/top_offers['Customer ID'].sum()
    top_offers_min = top_offers[top_offers['perc_total']>minimal_threshold].head(max_offers_to_return)
        
    return top_offers_min['offer'].unique()

# %% [markdown]
# #### **Function Overview**
# 
# This function finds n similar customers with respect to the distance metric provided by user from training data corresponding to specific customer. 
# 
# It further finds the most occurring offers given to set of n similar customers and recommend the top 3 offers.
# 
# 
# ##### **Input Parameters**
# 1. df: The main DataFrame that contains all the data for all customers and their offers
# 
# 2. df_id: A DataFrame that contains only the customer ID, month of joining, and churn value and offer
# 
# 3. customer_id: The ID of the customer we want to recommend offers for
# 
# 4. month: The month we want to make the offer for
# 
# 5. distance_func: The distance function to use for finding similar customers
# 
# 6. n: The number of other customers we want to base our recommendations on
# 
# 7. minimal_threshold: The minimal threshold of prevalence of a given offer in the similar group of customers, for it to be considered for recommendation
# 
# 8. max_offers_to_return: The maximum number of offers to return
# 
# 

# %% [markdown]
# ##### **Step by Step Explanation**
# 
# * Extracts the feature vectors of all customers by removing the 'Customer ID', 'Month', 'Month of Joining', and 'offer' columns from the main DataFrame
# 
# ```
# features = list(df.columns.difference(['Customer ID','Month','Month of Joining','offer']))
# X = df[features].values
# ```
# 
# * Extracts the feature vector of the given customer
# 
# ```
# index = df[(df['Customer ID'] == customer_id) & (df['Month']==month)].index[0]
# x = X[index]
# ```
# 
# * Computes the distances between the feature vectors of the given customer and all other customers using the specified distance function
# 
# ```
# if distance_func == 'euclidean':
#     distances = euclidean_distances(X, x.reshape(1, -1)).flatten()
# elif distance_func == 'manhattan':
#     distances = manhattan_distances(X, x.reshape(1, -1)).flatten()
# elif distance_func == 'cosine':
#      distances = 1 - cosine_similarity(X, x.reshape(1, -1)).flatten()
# else:
#     raise ValueError('Invalid distance function specified.')
# ```
# 
# * Finds the indices of the n customers with the lowest distance to the given customer
# 
# ```
# most_similar_indices = distances.argsort()[:n]
# ```
# 
# * Extracts the customer data for the most similar customers
# 
# ```
# similar_customers = df.iloc[most_similar_indices]
# ```
# 
# * Merges the similar customers DataFrame with the ID DataFrame to select only the customers who did not churn
# 
# ```
# similar_customers = pd.merge(similar_customers,df_id[['Customer ID','Month of Joining','Month','Churn Value']],on=['Customer ID','Month of Joining','Month','Churn Value'])
# ```
# 
# * Selects the customers that did not churn
# 
# ```
# similar_customers = similar_customers[similar_customers['Churn Value']==0]
# ```
# 
# * Counts the top offers of the non-churned customers, calculates the percentage of each offer among the top offers, and selects the top max_offers_to_return offers whose percentage is above the minimal_threshold
# 
# ```
# top_offers = similar_customers[['Customer ID','offer']].groupby(['offer']).agg({'Customer ID':'count'}).reset_index().sort_values(by = 'Customer ID', ascending = False)
# top_offers['perc_total'] = top_offers['Customer ID']/top_offers['Customer ID'].sum()
# top_offers_min = top_offers[top_offers['perc_total']>minimal_threshold].head
# ```

# %% [markdown]
# ## **The minimal threshold parameter**
# 
# Whenever we are building any framework, and especially unsupervised ones, it is important that we establish parameters that can help us have confidence in what we are doing.
# 
# In this particular case, we are assigning a minimal threshold of 10% for an offer to be potentially chosen to customers.
# 
# This comes from the fact that we have 10 offers (A-> J). If we were to randomly assign an offer to a customer, we would likely give each offer an equal probability of being assigned, so 100%/10%. So given that, if we are going to recommend something, it needs to be better than the random assignment.

# %% [markdown]
# To identify similar customers, we are going to treat each feature that we selected now in the train dataframe as a customer feature

# %% [markdown]
# ## **Which distance to choose?**
# 
# In Summary
# *  Manhattan distance: This metric measures the distance between two points by summing the absolute differences between their coordinates. It is also called the "taxicab" or "city block" distance because it measures the distance a taxicab would have to travel to get from one point to another on a city grid. 
# 
# *  Cosine similarity: This metric measures the cosine of the angle between two vectors in a high-dimensional space. It is commonly used in text analysis and information retrieval to measure the similarity between documents. Cosine similarity is often preferred over Euclidean distance when the magnitude of the vectors is not important, and only the direction matters.
# 
# *  Euclidean distance: This metric measures the distance between two points in a straight line. It is the most common distance metric used in machine learning and data science. Euclidean distance is useful when the data is dense, and the features have similar scales.
# 
# In general, if you have high-dimensional data or sparse data, Manhattan or cosine distance may be more appropriate. If you have dense data with similar scales, Euclidean distance is a good choice.

# %% [markdown]
# ## **Applying this framework to a specific customer**

# %%
train[train['Customer ID']=='sirifvlkipkel21']

# %%
customer_id = 'sirifvlkipkel21' # This Customer Id needs to be there from the training dataset
month = 12
distance_func = 'euclidean'
n = 1000
minimal_threshold= 0.10
max_offers_to_return = 3
id_cols=['Customer ID','Month','Month of Joining','Churn Value','offer']

offers = get_recommended_offers(train, train[id_cols], customer_id,month,distance_func,n,minimal_threshold,max_offers_to_return)

print('The first offer to recommend is ' + str(offers[0]))
print('The second offer to recommend is ' + str(offers[1]))
print('The third offer to recommend is ' + str(offers[2]))

# %% [markdown]
# Instead of adding this result to a list, we could also add it to a dataframe:

# %%
frame = pd.DataFrame()

data = {'Customer ID': [customer_id],
        'Month': [month],
        'offer 1': [str(offers[0])],
        'offer 2': [str(offers[1])],
        'offer 3': [str(offers[2])]}

frame= pd.DataFrame(data)
frame.head(2)

# %% [markdown]
# ## **Bootstrapping the framework**
# 
# 
# Especially in unsupervised learning problems, it is always a good idea to run several approaches ('bootstrap') and chose the most common answer amongst the different models. This mechanism is similar to what algorithms like random forest do, for example: they fit several trees and each tree votes the final classification of a sample.
# 
# We are going to play with the 3 distances we have in our function + the number of customers we pull the data from in order to get a voted answer

# %%
def find_similar_customers_multiple(df:pd.DataFrame, df_id:pd.DataFrame,customer_id:str,month:int,distance_funcs:list,n_values,minimal_threshold:float,max_offers_to_return:int):
    """
    Given a dataframe, a customer_id, n values, and distance functions,
    run multiple iterations of the find_similar_customers function with different parameter combinations,
    and return the top 3 most common answers among those.
    """
    results = []
    for n in n_values:
      for distance_func in distance_funcs:
          result = get_recommended_offers (df,df_id ,customer_id,month,distance_func,n,minimal_threshold,max_offers_to_return)
          results.append(result)
          # concatenate the arrays together
          concatenated_array = np.concatenate(results)
          # convert the concatenated array to a Python list
          result_list = list(concatenated_array)
          result_list
    if len(results) == 0:
        return None
    else:
        result_counts = pd.Series(result_list).value_counts()
        most_common_result = [result_counts.index[0],result_counts.index[1],result_counts.index[2]]
        return most_common_result

# %% [markdown]
# #### **Function Overview**
# 
# This function takes a dataframe, a customer ID, a month, a list of distance functions, a list of n values, a minimal threshold, and a max number of offers to return as input parameters. It uses these parameters to run multiple iterations of the get_recommended_offers function and returns the top 3 most common recommendations among them.
# 
# First, the function initializes an empty results list to store the results of each iteration. Then, for each n value and distance function, it calls the get_recommended_offers function with the given parameters and appends the result to the results list. After all iterations are completed, the function concatenates the arrays together and converts them to a Python list.
# 
# If the results list is empty, meaning no recommendations were generated, the function returns None. Otherwise, it uses the value_counts() function to count the frequency of each recommended offer in the result_list, and returns a list of the top 3 most common recommendations.
# 
# 
# 
# 
# 
# 

# %% [markdown]
# ## **Applying this framework to a given customer** ##

# %%
n_values = [100, 250, 500, 1000]
distance_funcs = ['euclidean', 'manhattan', 'cosine']
customer_id = 'sirifvlkipkel21'
month = 12
minimal_threshold= 0.10
max_offers_to_return = 3
id_cols=['Customer ID','Month','Month of Joining','Churn Value','offer']
find_similar_customers_multiple(train, train[id_cols],customer_id,month,distance_funcs,n_values,minimal_threshold=0.10,max_offers_to_return=3)

# %% [markdown]
# ### Seeing how this function work in details

# %%
# parameter values
n_values = [100, 250, 500, 1000]
distance_funcs = ['euclidean', 'manhattan', 'cosine']
customer_id = 'sirifvlkipkel21'
month = 12
minimal_threshold= 0.10
max_offers_to_return = 3
results = []


# %%
# for each value in n_values 
# The number of other customers we want to base our recommendations on
for n in n_values:
  # chose distance metric from a list
  for distance_func in distance_funcs:
    
      # get the offer recommendation by using get_recommended_offers function
      result = get_recommended_offers (train, train[id_cols],customer_id,month,distance_func,n,minimal_threshold,max_offers_to_return)
      
      results.append(result)
      # concatenate the arrays together
      concatenated_array = np.concatenate(results)
      # convert the concatenated array to a Python list
      result_list = list(concatenated_array)
if len(results) == 0:
    None
else:
    result_counts = pd.Series(result_list).value_counts()
    most_common_result = [result_counts.index[0],result_counts.index[1],result_counts.index[2]]
    most_common_result

# %%
result_counts 

# %% [markdown]
# We can see above that the different iterations returned 7 potential offers. 3 of those were, however, much more common than 6 of the rest.

# %%
most_common_result

# %% [markdown]
# The most_common_result list will aggregate the top 3 offers

# %% [markdown]
# ## **Applying this to the whole dataframe**

# %% [markdown]
# This code defines a function called production_model that takes three arguments: df, distance_func, and n. df is a Pandas DataFrame that contains data about customer transactions. The distance_func is a function that takes two arguments and returns a distance measure between them. The n parameter specifies the number of recommended offers to return for each customer and month.
# 
# The function initializes an empty DataFrame called frame and then iterates over each unique customer in the input DataFrame. For each customer, the function iterates over each unique month for that customer and calls another function called get_recommended_offers. This function returns a list of recommended offers for that customer and month based on their transaction history, using the distance_func and n parameters.
# 
# The function then creates a new DataFrame called frame1 containing the Customer ID, Month, and the top three recommended offers for that customer and month, and appends it to the frame DataFrame. Finally, the function returns the frame DataFrame containing the recommendations for all customers and months in the input DataFrame.

# %%
train.head(2)

# %% [markdown]
# ### **For the function without bootstrapping**

# %%
train_id.columns

# %%
def production_model(df, train, production, distance_func, n):
    frame = pd.DataFrame()
    # Pour chaque client et chaque mois
    for customer in list(df['Customer ID'].unique()):
        for month in list(df[df['Customer ID'] == customer]['Month'].unique()):
            # Cette partie du code ajoute la ligne que nous voulons obtenir des offres à l'ensemble d'entraînement, afin que nous puissions utiliser la formule de distance
            data = pd.DataFrame()
            data = pd.concat([train, production[(production['Customer ID'] == customer) & (production['Month'] == month)]])
            data = data.reset_index()
            data_id = data[['Customer ID', 'Month', 'Month of Joining', 'Churn Value']]
            results = get_recommended_offers(data, data_id, customer, month, distance_func, n, minimal_threshold=0.10, max_offers_to_return=3)
            data = {'Customer ID': [customer],
                    'Month': [month],
                    'offers': [results]}
            frame1 = pd.DataFrame(data)
            frame = frame._append([frame1])
    return frame

# %% [markdown]
# #### **Function Overview**
# 
# The production_model function takes in a dataframe df, a distance function distance_func, and a value n, and returns a new dataframe frame with recommended offer in each month.
# 
# For each unique Customer ID and Month combination in the input dataframe, the function adds the line for that combination to a new dataframe data that includes all previous training data plus the current combination. The get_recommended_offers function is then called on this new data dataframe, using the given distance_func and n values, to obtain a list of recommended offers for that combination.
# 
# A dictionary is then created to store the Customer ID, Month, and offers data, and a new dataframe frame1 is created from this dictionary. This new dataframe is then appended to the existing frame dataframe.
# 
# Finally, the frame dataframe is returned, which contains the recommended offers for each unique Customer ID and Month combination in the input dataframe.

# %%
frame_production_100_samples = production_model(production.head(100),train=train,production=production,distance_func = 'euclidean',n = 250)
frame_production_100_samples.to_csv('./data/output/offer_recommendation_without_bootstap.csv',index=False)

# %%
frame_production_100_samples.head(5)

# %% [markdown]
# ### **For the function with bootstrapping**

# %%
def production_model_bootstrap(df, train, production, distance_funcs, n_values):
    frame = pd.DataFrame()
    for customer in list(df['Customer ID'].unique()):
        for month in list(df[df['Customer ID'] == customer]['Month'].unique()):
            # Cette partie du code ajoute la ligne que nous voulons obtenir des offres à l'ensemble d'entraînement, afin que nous puissions utiliser la formule de distance
            data = pd.DataFrame()
            data = pd.concat([train, production[(production['Customer ID'] == customer) & (production['Month'] == month)]])
            data = data.reset_index()
            data_id = data[['Customer ID', 'Month', 'Month of Joining', 'Churn Value']]
            results = find_similar_customers_multiple(data, data_id, customer, month, distance_funcs=distance_funcs, n_values=n_values, minimal_threshold=0.10, max_offers_to_return=3)
            data = {'Customer ID': [customer],
                    'Month': [month],
                    'offers': [results]}
            frame1 = pd.DataFrame(data)
            frame = frame._append(frame1)
    return frame


# %% [markdown]
# #### **Function Overview**
# 
# The production_model function takes in a dataframe df, a distance function distance_func, and a value n, and returns a new dataframe frame with recommended offer in each month.
# 
# For each unique Customer ID and Month combination in the input dataframe, the function adds the line for that combination to a new dataframe data that includes all previous training data plus the current combination. The find_similar_customers_multiple(boostraping function) function is then called on this new data dataframe, using the given distance_func and n values, to obtain a list of recommended offers for that combination.
# 
# A dictionary is then created to store the Customer ID, Month, and offers data, and a new dataframe frame1 is created from this dictionary. This new dataframe is then appended to the existing frame dataframe.
# 
# Finally, the frame dataframe is returned, which contains the recommended offers for each unique Customer ID and Month combination in the input dataframe.

# %%
frame_production_100_samples_bootstrap = production_model_bootstrap (production.head(100),train=train,production=production,distance_funcs=['euclidean', 'manhattan', 'cosine'],n_values=[250,500,1000])
frame_production_100_samples_bootstrap.to_csv('./data/output/offer_recommendation_bootstrap.csv',index=False)


# %%
frame_production_100_samples_bootstrap.head(5)

# %% [markdown]
# ## **Questions**
# 
# 
# How would I validate this approach?
# 
# Ideally, I could design a test where, for some control group, I offer a random offer (or use whatever method is used today). For some treatment group, I use my model to assign the offer. I could then measure retention of these two groups.
# 
# Another option would be to train my model on part of the training set and use another part of the training set to test it. I could check if I have recommended a 'winning offer,' where 'winning' means an offer that was accepted by the customer and prevented churn in the next month.
# 

# %% [markdown]
# ## **Conclusion**
# 
# In this project, I used a simple yet effective unsupervised model to provide offers to customers.
# 
# A collaborative offer recommendation system can be a valuable tool for telecom companies like mine to increase customer satisfaction and revenue. By using data on customer behavior and preferences, our system can provide personalized offers that are more likely to be accepted by our customers.
# 
# To implement such a system, I had to take several steps, including collecting and cleaning data, creating customer profiles, selecting appropriate distance functions, and validating the system's performance. I also had to consider ethical considerations related to data privacy and security.
# 
# Overall, a collaborative offer recommendation system can be a powerful tool for my telecom company to enhance our marketing strategies and provide better services to our customers. However, it is important for me to continuously evaluate and update the system to ensure its effectiveness and address any potential issues.
# 
# In conclusion, as a data scientist, I've learned that a successful data science project requires a clear understanding of the business problem and the data available, as well as the ability to select and apply appropriate data preprocessing techniques, feature engineering methods, and machine learning algorithms. It is also important for me to assess and optimize the performance of the model and communicate the results effectively to stakeholders.


