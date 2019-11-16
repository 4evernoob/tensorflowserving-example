import numpy as np
import random
import tensorflow as tf
import os
np.set_printoptions(precision=3)
# this model was created on the fly sorry for any inconvenience and the bad solution to this problem
#load dataset
var =np.loadtxt(open("wine.data", "rb"), delimiter=",")
random.shuffle(var)
# features
x=var[:-5,1:]
x=tf.keras.utils.normalize(x,axis=-1,order=2)

#tags
y=np.add(var[:-5,0],-1)
#convert to categorical
y = tf.keras.utils.to_categorical(y,3)
xt=var[-5:,1:]
xt=tf.keras.utils.normalize(xt,axis=-1,order=2)
yt=np.add(var[-5:,0],-1)
#convert to categorical
yt = tf.keras.utils.to_categorical(yt,3)
print(xt,yt)
print(np.shape(x[0]))

#model definition
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24,input_shape=np.shape(x[0])),
    tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(8 ),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
#model training
model.fit(x=x, y=y,batch_size=64,  epochs=1000, validation_split=0.2)
# just watching some results
print(model.predict(xt))
print(yt)

# we export model to be served
print('saving model')
model.save(os.path.join('modeldeploy','eltesto','1'))

