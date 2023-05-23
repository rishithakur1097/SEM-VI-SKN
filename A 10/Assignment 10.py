#Data Visualization 3
#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#read dataset
df = pd.read_csv('C:\\Users\\ATHARAV\\Desktop\\DSBDA Codes\\Assignment 10\\iris.csv')
print('Iris dataset is successfully loaded into the Dataframe......')

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


#Histogram Of 1-Variables
fig,axes = plt.subplots(2,2)
fig.suptitle('Histogram Of 1-Variable')
sns.histplot(data = df,x ='sepal.length' ,ax = axes[0,0])
sns.histplot(data = df,x ='sepal.width' ,ax = axes[0,1])
sns.histplot(data = df,x ='petal.length' ,ax = axes[1,0])
sns.histplot(data = df,x ='petal.width' ,ax = axes[1,1])
plt.show()

#Histogram Of 2-Variables
fig,axes = plt.subplots(2,2)
fig.suptitle('Histogram Of 2-Variables ')
sns.histplot(data = df,x ='sepal.length',hue='variety',multiple='dodge',ax = axes[0,0])
sns.histplot(data = df,x ='sepal.width',hue='variety',multiple='dodge',ax = axes[0,1])
sns.histplot(data = df,x ='petal.length',hue='variety',multiple='dodge',ax = axes[1,0])
sns.histplot(data = df,x ='petal.width',hue='variety',multiple='dodge',ax = axes[1,1])
plt.show()


#Boxplot Of 1-Variables
fig,axes = plt.subplots(2,2)
fig.suptitle('Boxplot Of 1-Variable ')
sns.boxplot(data = df,x ='sepal.length' ,ax = axes[0,0])
sns.boxplot(data = df,x ='sepal.width' ,ax = axes[0,1])
sns.boxplot(data = df,x ='petal.length' ,ax = axes[1,0])
sns.boxplot(data = df,x ='petal.width' ,ax = axes[1,1])
plt.show()

#Histogram Of 2-Variables
fig,axes = plt.subplots(2,2)
fig.suptitle('Boxplot Of 2-Variables ')
sns.boxplot(data = df,x ='sepal.length',y='variety',hue='variety',ax = axes[0,0])
sns.boxplot(data = df,x ='sepal.width',y='variety',hue='variety',ax = axes[0,1])
sns.boxplot(data = df,x ='petal.length',y='variety',hue='variety',ax = axes[1,0])
sns.boxplot(data = df,x ='petal.width',y='variety',hue='variety',ax = axes[1,1])
plt.show()
