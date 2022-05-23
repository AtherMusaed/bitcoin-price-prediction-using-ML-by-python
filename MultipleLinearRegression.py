# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:47:02 2022

@author: athee
"""
import pandas as pd
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
    

df = pd.read_csv('Bitcoin.csv')

#deletearange of rowS-index values لين 2018
df= df.drop(labels=range (0, 1202), axis=0)
df= df.reset_index()
#drop currency column
df.drop('Currency',axis=1 ,inplace= True)
df= df.drop(['index'], axis=1)
#change the date format 
df['Date'] = pd.to_datetime(df.Date)

#show data info
df.info()

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History', fontsize=25)
plt.plot(df['Close'])
plt.xlabel('Days since 2018', fontsize=20)
plt.ylabel('Close Price USD ($)', fontsize=20)
plt.show()

#split data into dependent and independet variables 
x= df[['Open','High','Low','Volume']]
y= df['Close']

#divided the dataset into testing and training 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)

#LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#show the coefficient of the reg
print(regressor.coef_)
#intercept
print(regressor.intercept_)
#predicct 
predicted=regressor.predict(x_test)
print(x_test)

#create another dataframe 
df=pd.DataFrame(y_test,predicted)
dfr=pd.DataFrame({'Actual Price':y_test, 'Predicted Price':predicted})
print(dfr)


errors = abs(y_test  - predicted)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# visul the bar chart
graph=dfr.head(20)
graph.plot(kind='bar')


