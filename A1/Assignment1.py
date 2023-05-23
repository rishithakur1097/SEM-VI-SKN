#Data Wranging1
#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#read dataset
#df = pd.read_csv('C:\Users\ATHARAV\Desktop\DSBDA Codes\Assignment 1\placement_data.csv')
df = pd.read_csv('C:\\Users\\ATHARAV\\Documents\\placement_data.csv')
print('Placement dataset is successfully loaded into the Dataframe......')

#Display information of dataset
print('Information of datset :\n',df.info)
print('Shape of Dataset (row X column): ',df.shape)
print('Columns Name : ',df.columns)
print('Total Elements in datset : ',df.size)
print('Datatype of Attributes (Columns):',df.dtypes)
print('First 5 Rows:\n',df.head().T)
print('Last 5 Rows:\n',df.tail().T)
print('Any 5 Rows:\n',df.sample(5).T)

#Display Statistical Information of Dataset
print('Statistical Information of Numerical Columns :\n',df.describe)

#Display Null Values
print('Total Number Of Null Values in Dataset : \n',df.isna().sum())

#Data Type Conversion
print('Converting Data Types of Variables : ')
df['sl_no'] = df['sl_no'].astype('int8')
print('Check Datatype of sl_no',df.dtypes)
df['ssc_p'] = df['ssc_p'].astype('int8')
print('Check Datatype of ssc_p',df.dtypes)


#Label Encoding conversion of Categorical to Quantitative
print('Encoding using Label Encoding (Cat codes)')
df['gender'] = df['gender'].astype('category')
print('Data Types of Gender= ',df.dtypes['gender'])
df['gender'] = df['gender'].cat.codes
print('Data types of gender after label encoding = ',df.dtypes['gender'])
print('Gender Values :',df['gender'].unique())

#Normalisation
print('Normalisation using Min-Max Feature Scalling: ')
df['salary'] = (df['salary'] - df['salary'].min())/(df['salary'].max() - df['salary'].min())
print(df.head().T)
  

