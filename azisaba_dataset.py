import os
import cv2
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array,load_img

def azisaba_dataset():
    train_data = []
    train_label = []
    azidirPath = './gendata/azi'
    sabadirPath = './gendata/saba'
    azifileList = os.listdir(azidirPath)
    sabafileList = os.listdir(sabadirPath)
    for azifile in azifileList:
        azifilename = azidirPath + '/' +azifile
        temp_img = load_img(azifilename, target_size=(250,75))
        temp_img_array  = img_to_array(temp_img)
        train_data.append(temp_img_array)
        train_label.append(0)
    
    for sabafile in sabafileList:
        sabafilename = sabadirPath + '/' +sabafile
        temp_img = load_img(sabafilename, target_size=(250,75))
        temp_img_array  = img_to_array(temp_img)
        train_data.append(temp_img_array)
        train_label.append(1)
    
    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    return (train_data,train_label)

def azisaba_test():
    test_data = []
    test_label = []
    azidirPath = './data/azi'
    sabadirPath = './data/saba'
    azifileList = os.listdir(azidirPath)
    sabafileList = os.listdir(sabadirPath)
    for azifile in azifileList:
        azifilename = azidirPath + '/' +azifile
        temp_img = load_img(azifilename, target_size=(250,75))
        temp_img_array  = img_to_array(temp_img)
        test_data.append(temp_img_array)
        test_label.append(0)
    
    for sabafile in sabafileList:
        sabafilename = sabadirPath + '/' +sabafile
        temp_img = load_img(sabafilename, target_size=(250,75))
        temp_img_array  = img_to_array(temp_img)
        test_data.append(temp_img_array)
        test_label.append(1)
    
    test_data = np.asarray(test_data)
    test_label = np.asarray(test_label)
    return (test_data,test_label)