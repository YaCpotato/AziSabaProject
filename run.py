import os
import glob
import numpy as np
import azisabamodel
import azisaba_dataset
from keras.utils import np_utils
import matplotlib.pyplot as plt
(X_train,Y_train) = azisaba_dataset.azisaba_dataset()
print(X_train.shape,Y_train.shape)
Y_train = np_utils.to_categorical(Y_train, 2)
model = azisabamodel.azisabamodel()
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.summary()
model.fit(X_train, Y_train, batch_size=16, epochs=1,verbose=1,validation_split=3.0,shuffle=True)

(X_test,Y_test) = azisaba_dataset.azisaba_test()
Y_test = np_utils.to_categorical(Y_test, 2)
model.evaluate(X_test,Y_test,verbose=1)