#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 00:25:53 2021

@author: mohamad
"""
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
class Help(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)

        loadUi("help.ui",self)
if __name__=="__main__":
    
    app = QApplication([])
    window = Help()
    window.show()
    app.exec_()