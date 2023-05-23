#Data Wranging2

def RemoveOutlier(df,var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high,low = Q3 + 1.5*IQR , Q1 - 1.5*IQR
    df = df[((df[var]>=low) & (df[var]<= high))]
    print('Outliers removed in ',var)
    return df

def DisplayOutliers(df,message):
    fig,axes = plt.subplots(2,2)
    fig.suptitle(message)
    sns.boxplot(data = df,x ='raisedhands' ,ax = axes[0,0])
    sns.boxplot(data = df,x ='VisITedResources' ,ax = axes[0,1])
    sns.boxplot(data = df,x ='AnnouncementsView' ,ax = axes[1,0])
    sns.boxplot(data = df,x ='Discussion' ,ax = axes[1,1])
    fig.tight_layout()
    plt.show()

#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#read datset
df = pd.read_csv('C:\\Users\\ATHARAV\\Desktop\\DSBDA Codes\\Assignment 2\\student_data.csv')
print('Student Academic Performance dataset is successfully loaded into the Dataframe......')

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

#Handling Outliers
DisplayOutliers(df,'Before Removing Outliers')
df = RemoveOutlier(df,'raisedhands')
df = RemoveOutlier(df,'VisITedResources')
df = RemoveOutlier(df,'AnnouncementsView')
df = RemoveOutlier(df,'Discussion')
DisplayOutliers(df,'After Removing Outliers')

#Conversion of Categorical to Quantitative(Encoding)
df['gender'] = df['gender'].astype('category')
df['gender'] = df['gender'].cat.codes
print('Data types of gender after label encoding = ',df.dtypes['gender'])
print('Gender Values :',df['gender'].unique())

sns.boxplot(data=df,x='gender',y='raisedhands',hue='gender')
plt.title('Box Plot with 2 variables Gender And Raisedhands')
plt.show()


sns.boxplot(data=df,x='NationalITy',y='Discussion',hue='gender')
plt.title('Box Plot with 3 variables Nationality,Discussion,Gender')
plt.show()

print('Realtionship between variables using Scatterplot: ')
sns.scatterplot(data = df,x='raisedhands',y='VisITedResources')
plt.title('Scatter plot for Visited Resources,Raisedhands')
plt.show()





