#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:33:30 2021

@author: mohamad
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:52:54 2020

@author: mohamad
"""


from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random
import time    


from paintwidget import Window
from Help import Help
# example of a cnn for image classification
from numpy import asarray
from numpy import unique
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import load_model

class Mlp(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)

        loadUi("dising.ui",self)
        self.paint.clicked.connect(self.open_paint)  
        self.file.clicked.connect(self.open_file)
        
        self.Train.pressed.connect(self.consolclick)
        self.Train.clicked.connect(self.Train_class)
        self.test.clicked.connect(self.test_file)
        self.pushButton.clicked.connect(self.say_help)
        self.test.pressed.connect(self.consolclick3)
        self.consol.append("Cnn Classification.....")
    def say_help(self) :
        
        self.ui=Help();
        self.ui.show()
    def consolclick(self):
        self.consol.append("clicked Train Waitin ....")
    def consolclick3(self):
        self.consol.clear()
        self.consol.append("clicked Test Waitin ....")
    def Train_class(self):
        
        y,x=self.extrac()
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.12, random_state=10)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        in_shape = X_train.shape[1:]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)
        n_classes = len(unique(y_train))
        number1=self.lineEdit_noron2.text()
        activ1=self.lineEdit_activ.text()
        n_kernel1=self.lineEdit_4.text()
        kernel1=self.lineEdit_kernel1.text()
        ######
        number2=self.lineEdit_noron1.text()
        activ2=self.lineEdit_active2.text()
        n_kernel2=self.lineEdit_n_kernel.text()
        kernel2=self.lineEdit_kernel.text()
        max1=self.lineEdit_max.text()
        print(in_shape, n_classes)
        model = Sequential()
        model.add(Conv2D(int(number1), (int(n_kernel1),int(n_kernel1)), activation=activ1, kernel_initializer=kernel1, input_shape=in_shape))
        model.add(Conv2D(int(number2), (int(n_kernel2),int(n_kernel2)), activation=activ2, kernel_initializer=kernel2, input_shape=in_shape))
        model.add(MaxPool2D((int(max1), int(max1))))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history=model.fit(X_train, y_train, epochs=20,batch_size=128,validation_split=0.1)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        accurecy='Accuracy Test: '+ str(acc)
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.consol.append(short_model_summary)
        self.consol.append(accurecy)
        
        self.widget.canvas.axes.plot( history.history['accuracy'], label='train')
        self.widget.canvas.axes.plot( history.history['val_accuracy'], label='val')
        self.widget.canvas.axes.set_xlabel("Epoch")
        self.widget.canvas.axes.set_ylabel("accuracy")
        self.widget.canvas.axes.legend()
        self.widget.canvas.draw()
        model.save('model.h5')
        
    def open_file(self):
        
        dir_path=QFileDialog.getExistingDirectory(None,"Choose Directory","E:\\",QFileDialog.ShowDirsOnly)
        print(dir_path)
        model = load_model("model.h5")
        i=0
        for filename in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path,filename),cv2.IMREAD_GRAYSCALE)
            
            t = np.array(img)
            t = t.astype(float) / 255.
            t=cv2.resize(t,(28,28))
            t = t.reshape( t.shape[0] , t.shape[1],1)
            yhat = model.predict(asarray([t]))
            
            if argmax(yhat)==0:
                cv2.imshow(str(i) +" :predict A",img)
            if argmax(yhat)==1:
                cv2.imshow(str(i) +" :predict B",img)
            if argmax(yhat)==2:
                cv2.imshow(str(i) +" :predict D",img)

            if argmax(yhat)==3:
                cv2.imshow(str(i) +" :predict G",img)      
            i+=1
        self.consol.clear()
        self.consol.append("show image titel pridect")
        
        
    def open_paint(self):         
         self.ui=Window();
         self.ui.show()
         self.test.setEnabled(True)
         
    def test_file(self):
        img = cv2.imread("test.png",cv2.IMREAD_GRAYSCALE)
        model = load_model("model.h5")
        img = np.array(img)
        img = img.astype("float32") / 255.
        img=cv2.resize(img,(28,28))
        img = img.reshape( img.shape[0] ,img.shape[1],1)
        print(img.shape)
        yhat = model.predict(asarray([img]))
        y=argmax(yhat)
        self.consol.clear()
        if(y==0):
             self.consol.append("peridict drow paint A")
        if(y==1):
             self.consol.append("peridict drow paint B")
        if(y==2):
             self.consol.append("peridict drow paint D")
        if(y==3):
             self.consol.append("peridict drow paint G")
    
            
            
            
    def extrac(self):
        images=[]
        label=[]
        for filename in os.listdir("Dataset/A"):
            img = cv2.imread(os.path.join("Dataset/A",filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img,(28,28)))
                label.append(1)
        for filename in os.listdir("Dataset/B"):
            img = cv2.imread(os.path.join("Dataset/B",filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img,(28,28)))
                label.append(2)
        for filename in os.listdir("Dataset/D"):
            img = cv2.imread(os.path.join("Dataset/D",filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img,(28,28)))
                label.append(3)
        for filename in os.listdir("Dataset/j"):
            img = cv2.imread(os.path.join("Dataset/j",filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img,(28,28)))
                label.append(4)
        images = np.array(images)
        
        
        return label ,images    
if __name__=="__main__":
    
    app = QApplication([])
    window = Mlp()
    window.show()
    app.exec_()