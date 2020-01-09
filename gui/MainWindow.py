# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:12:24 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

from PyQt5.QtWidgets import (QMainWindow,QTabWidget,QWidget,QVBoxLayout,QPushButton,
                QTableWidget,QTableWidgetItem,QHeaderView)

class tabLayout(QWidget):
    def __init__(self,parent,par):
        super(QWidget,self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.par = par
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        #self.tabs.resize(300,200)
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Capture")
        self.tabs.addTab(self.tab2,"Parameters")
        
        # Create first tab [dummy]
        self.tab1.layout = QVBoxLayout(self)
        self.pushButton1 = QPushButton("PyQt5 button")
        self.tab1.layout.addWidget(self.pushButton1)
        self.tab1.setLayout(self.tab1.layout)
        
        # Create second tab [parameters]
        self.tab2.layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setRowCount(len(self.par))
        self.table.setHorizontalHeaderLabels(('name','value','type'))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(False)
        # Populate table
        for row,key in enumerate(self.par.keys()):
            newkey = QTableWidgetItem(key)
            newitem = QTableWidgetItem(str(self.par[key]))
            newtype = QTableWidgetItem(type(self.par[key]).__name__)
            self.table.setItem(row,0,newkey)
            self.table.setItem(row,1,newitem)
            self.table.setItem(row,2,newtype)

        self.tab2.layout.addWidget(self.table)
        self.tab2.setLayout(self.tab2.layout)
        
        
        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        

class MainWindow(QMainWindow):
    def __init__(self,par):
        super().__init__()
        self.setup(par)
        
    
    def setup(self,par):
        self.setWindowTitle('pySFDI: an Uber SFDI GUI')
        self.tabWidget = tabLayout(self,par)
        
        self.setCentralWidget(self.tabWidget)
        
        self.show()