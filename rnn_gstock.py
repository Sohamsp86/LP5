import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,SimpleRNN
from tensorflow.keras.models import Sequential

df=pd.read_csv("Google_Stock_Price_Train.csv")
df
data=df['Open'].values.reshape(-1,1)

s=MinMaxScaler(feature_range=(0,1))
data=s.fit_transform(data)

xtrain=[]
ytrain=[]
for i in range(60,len(data)):
    xtrain.append(data[i-60:i,0])
    ytrain.append(data[i,0])

xtrain,ytrain=np.array(xtrain),np.array(ytrain)
xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)

model=Sequential() 
model.add(SimpleRNN(units=50,activation='tanh',return_sequences=False,input_shape=(60,1)))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(xtrain,ytrain,epochs=20,batch_size=32)

ypred=model.predict(xtrain)
ypred=s.inverse_transform(ypred)
ytrain=s.inverse_transform(ytrain.reshape(-1,1))

plt.figure(figsize=(10,10),)
plt.plot(ypred,label='Predicted')
plt.plot(ytrain,label='Actual')
plt.legend()
plt.title('Google stock price')
plt.xlabel('Time')
plt.ylabel('Price')