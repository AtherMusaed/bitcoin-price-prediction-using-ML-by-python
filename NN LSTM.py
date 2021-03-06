# -*- coding: utf-8 -*-
"""
Created on Sun May 15 19:22:09 2022

@author: athee
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
 

df = pd.read_csv('Bitcoin.csv')

#deletearange of rowS-index values لين 2018
df= df.drop(labels=range (0, 1202), axis=0)
df= df.reset_index()
#drop currency column
df.drop('Currency',axis=1 ,inplace= True)
df= df.drop(['index'], axis=1)
#change the date format 
df['Date'] = pd.to_datetime(df.Date)

#Createanew dataframe with only the 'Close column
data =df.filter(['Close']) 
#Convert the dataframe toanumpy array
dataset=data.values
#Get the number of rows to train the model on 80% 
training_data_len=math.ceil( len(dataset)* .8)

#Scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
       
#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train data sets
x_train=[]
y_train=[]
for i in range(100, len(train_data)):
    x_train.append (train_data[i-100:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 100:
        print(x_train)
        print(y_train)



#Convert the x_train and y_train to numpy arrays
x_train, y_train=np.array(x_train), np.array(y_train)
#Reshape the data
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#Build the LSTM model
model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))
    
#compile the model 
model.compile(
optimizer='Adam' , loss='mean_squared_error',
              metrics=['accuracy'])
#train the model 
model.fit(x_train, y_train, batch_size=5, epochs=5)


#Create the testing data set
#Createanew array containing scaled values from range index 
test_data=scaled_data[training_data_len-100:, :]

#Create the data sets x_test and y_test
x_test=[] 
y_test=dataset[training_data_len:, :]
for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i, 0])

#Convert the data toanumpy array
x_test=np.array(x_test)
#Reshape the data
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
#Get the models predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
              
#Get the root mean squared error (RMSE)
rmse=np.sqrt( np.mean( predictions-y_test )**2)

#Plot the data
train =data[:training_data_len]
valid=data[training_data_len:]
valid[ 'Predictions']= predictions

#Visualize the data
plt.Figure(figsize =(16,8))
plt.title('prediction Model ',fontsize=25)
plt.xlabel('Date', fontsize=10)
plt.ylabel('Close Price USD ($)' , fontsize=10)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predictions)
cm

