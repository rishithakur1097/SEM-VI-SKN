#Data Analytics 1

def RemoveOutlier(df,var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high,low = Q3 + 1.5*IQR , Q1 - 1.5*IQR
    df = df[((df[var]>=low) & (df[var]<= high))]
    print('Outliers removed in ',var)
    return df

def DisplayOutliers(df,msg):
    fig,axes = plt.subplots(1,2)
    fig.suptitle(msg)
    sns.boxplot(data = df,x ='rm' ,ax = axes[0])
    sns.boxplot(data = df,x ='lstat' ,ax = axes[1])
    fig.tight_layout()
    plt.show()

#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#read datset
df = pd.read_csv('C:\\Users\\ATHARAV\\Desktop\\DSBDA Codes\\Assignment 4\\Boston.csv')
print('Boston dataset is successfully loaded into the Dataframe......')

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

#Finding Correaltion Matrix
print('Finding correlation matrix using heatmap')
sns.heatmap(df.corr(),annot = True)
plt.show()

#Finding And Removing Outliers
print('Finding And Removing Outliers')
DisplayOutliers(df,'Before Removing Outliers')
print('Identifying overall outliers in column name variables....')
df = RemoveOutlier(df,'lstat')
df = RemoveOutlier(df,'rm')
DisplayOutliers(df,'After Removing Outliers')

#split the data into inputs and outputs
x = df[['rm','lstat']] #input data
y = df['medv'] #output data

#Training and Testing Data
from sklearn.model_selection import train_test_split

#Assign Test Data Size 20%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#Apply linear regression model on training data
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train,y_train)
y_pred = model.predict(x_test)

#Display Accuracy Of the Model
from sklearn.metrics import mean_absolute_error
print('MAE:',mean_absolute_error(y_test,y_pred))
print('Model Score:',model.score(x_test,y_test))

#Test the model using user input
print('Predict House Price using user input: ')
features = np.array([[6,19]])
prediction = model.predict(features)
print('Prediction:{}',format(prediction))


