import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df=pd.read_csv('HousingData.csv')
df.dtypes
df.describe()
df=df.dropna()
df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
x=df[['LSTAT','RM']]
y=df['MEDV']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=0)
scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)
model=tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation='relu',input_shape=(xtrain.shape[1],)),
    tf.keras.layers.Dense(64,activation='relu',name='layer1'),
    tf.keras.layers.Dense(32,activation='relu',name='layer2'),
    tf.keras.layers.Dense(16,activation='relu',name='layer3'),
    tf.keras.layers.Dense(8,activation='relu',name='layer4'),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
model.fit(xtrain,ytrain,epochs=100,batch_size=32,validation_split=0.2,verbose=1)
ypred=model.predict(xtest)
mse=mean_squared_error(ytest,ypred)
mae=mean_absolute_error(ytest,ypred)
r2score=r2_score(ytest,ypred)
print(f"MSE={mse:.2f}")
print("MAE= %.2f"%mae)
print(f"R2={r2score:.2f}")
plt.figure(figsize=(10,10))
plt.scatter(ytest, ypred)
m,b = np.polyfit(ytest,ypred,1)
plt.plot(ytest,m*ytest+b)#,color='red',linewidth='2')
plt.show()