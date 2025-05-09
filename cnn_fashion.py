import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models

train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

X_train=train_df.drop('label',axis=1).values  
Y_train=train_df['label'].values

X_test=test_df.drop('label',axis=1).values
Y_test=test_df['label'].values

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),    #fully connected layer
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  #o/p layer
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=model.fit(X_train,Y_train, epochs=5,batch_size=32,validation_split=0.2)

loss,metrics=model.evaluate(X_test,Y_test)
print(metrics*100)

y_prob=model.predict(X_test)
print(y_prob)
#upper prob h har class ka 
#convert probability vectors into a single class label.
#argmax(axis=-1) returns the index of the highest value in each row
y_pred = y_prob.argmax(axis=-1)

#visualize 25 Fashion MNIST test images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10),)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])   #line hataaega axis waali
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i],plt.cm.binary)  #plt.cm.binary is a built-in colormap that shows images in black and white (grayscale). colormap
    plt.title(f"Pred:{class_names[y_pred[i]]}")
plt.show()
