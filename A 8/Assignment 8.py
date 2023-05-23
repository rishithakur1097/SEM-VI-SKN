#Data Visualization 1
#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#read dataset

df = pd.read_csv('C:\\Users\\ATHARAV\\Desktop\\DSBDA Codes\\Assignment 8\\titanic.csv')
print('Titanic dataset is successfully loaded into the Dataframe......')

#Display information of dataset
print('Information of datset :\n',df.info)
print('Shape of Dataset (row X column): ',df.shape)
print('Columns Name : ',df.columns)
print('Total Elements in datset : ',df.size)
print('Datatype of Attributes (Columns):',df.dtypes)
print('First 5 Rows:\n',df.head().T)
print('Last 5 Rows:\n',df.tail().T)
print('Any 5 Rows:\n',df.sample(5).T)

#Find Missing Values
print('Missing Values')
print(df.isnull().sum())

#Fill the missing Values
df['Age'].fillna(df['Age'].median(),inplace=True)
print('Null Values are: \n',df.isnull().sum())

#Histogram Of 1-Variables
fig,axes = plt.subplots(1,2)
fig.suptitle('Histogram Of 1-Variable (Age&Fare)')
sns.histplot(data = df,x ='Age' ,ax = axes[0])
sns.histplot(data = df,x ='Fare' ,ax = axes[1])
plt.show()

#Histogram Of 2-Variables
fig,axes = plt.subplots(2,2)
fig.suptitle('Histogram Of 2-Variables ')
sns.histplot(data = df,x ='Age',hue='Survived',multiple='dodge',ax = axes[0,0])
sns.histplot(data = df,x ='Fare',hue='Survived',multiple='dodge',ax = axes[0,1])
sns.histplot(data = df,x ='Age' ,hue='Sex',multiple='dodge',ax = axes[1,0])
sns.histplot(data = df,x ='Fare',hue='Sex',multiple='dodge',ax = axes[1,1])
plt.show()

