import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Flatten,Dense

df=pd.read_csv('imdb.csv')
df

df['sentiment']=df['sentiment'].map({'positive':1,'negative':0})
texts=df['review'].astype(str).tolist()
labels=df['sentiment'].tolist()

t=Tokenizer(num_words=10000,  oov_token='<OOV>')
t.fit_on_texts(texts)
sequences=t.texts_to_sequences(texts)
pad=pad_sequences(sequences,maxlen=256)

x=np.array(pad)
y=np.array(labels)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

model=Sequential([
    Embedding(input_dim=10000,output_dim=32),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(xtrain,ytrain,epochs=5,batch_size=64,validation_split=0.2)
model.evaluate(xtest,ytest)
newr=["It is amazing"]
seq=t.texts_to_sequences(newr)
padd=pad_sequences(seq,maxlen=256)
pred=model.predict(padd)[0][0]
if pred>0.5:
    print('positive')
else:
    print('negative')

ypred=model.predict(xtest)
ypred=(ypred>0.5).astype('int32')
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
print(cm)
sns.heatmap(cm,annot=True,fmt='d')

plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()