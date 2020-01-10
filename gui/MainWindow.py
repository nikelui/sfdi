# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:12:24 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys
from PyQt5.QtWidgets import (QMainWindow,QTabWidget,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,
                QTableWidget,QTableWidgetItem,QHeaderView,QLabel,QStyle,QSpacerItem,QSizePolicy)
from PyQt5.QtCore import Qt,QThread,pyqtSignal
from PyQt5.QtGui import (QPixmap,QImage,QFont)

sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import sfdi
import PyCapture2 as pc

class Thread(QThread):
    """Need to use thread, or it will freeze main window execution"""
    changePixmap = pyqtSignal(QImage)
    play = False
    
    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)       
        self.cameraSettings()

    def cameraSettings(self):
        self.cam = sfdi.setCamera_pg(num=0,res=self.par['res'],fps=self.par['fps']) # Set-up Camera
        try:
            self.cam.startCapture()
        except pc.Fc2error as fc2Err:
            print('Error starting capture: %s' % fc2Err)
    
    def run(self):
        while True:
            try:
                im = self.cam.retrieveBuffer()
            except pc.Fc2error as fc2Err:
                print('Error retrieving buffer: %s' % fc2Err)
            im = im.convert(pc.PIXEL_FORMAT.BGR) # from RAW to color (BGR 8bit)
            rawImage = QImage(im.data, im.shape[1], im.shape[0], QImage.Format_Indexed8)
            self.changePixmap.emit(rawImage)


class tabLayout(QWidget):
    
    def setPixMap(self, p):     
        p = QPixmap.fromImage(p)    
        p = p.scaled(640, 480, Qt.KeepAspectRatio)
        self.label.setPixmap(p)
    
    def __init__(self,parent,par,pdoc={}):
        super(QWidget,self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.par = par
        self.pdoc = pdoc
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        #self.tabs.resize(300,200)
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Capture")
        self.tabs.addTab(self.tab2,"Parameters")
        
        # Create first tab [dummy]
        self.tab1.layout = QHBoxLayout(self)
        self.tab1.layout.setContentsMargins(0,0,0,0)
        self.tab1.layout.setSpacing(0)
        ## Video goes here
        pix = QLabel(self)
        pix.setFixedSize(640,480)
        pix.setStyleSheet("background: red;")
        pix.setText("Video")
        pix.setFont(QFont('Calibri',30))
        
        ## Controls go here
        vlayout = QVBoxLayout(self)
        vlayout.setContentsMargins(0,0,0,0)
        vlayout.setSpacing(10)
        vlayout.setAlignment(Qt.AlignLeft)
        
        vSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
                
        hist = QLabel(self)
        hist.setFixedSize(160,100)
        hist.setStyleSheet("background: #7777FF")
        hist.setText("Histogram")
        hist.setFont(QFont('Calibri',20))
        vlayout.addWidget(hist)
        
        play = QPushButton()
        play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        play.setFixedSize(90,60)
        vlayout.addWidget(play)
        
        stop = QPushButton()
        stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        stop.setFixedSize(90,60)
        vlayout.addWidget(stop)
        
        
        start = QPushButton("Acquisition")
        start.setFont(QFont('Serif',20))
        start.setMinimumHeight(50)
        vlayout.addWidget(start)
        vlayout.addSpacerItem(vSpacer)
        
        self.tab1.layout.addWidget(pix,alignment=Qt.AlignTop)
        self.tab1.layout.addLayout(vlayout)
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
            newkey.setFlags(Qt.ItemIsEnabled)
            newitem = QTableWidgetItem(str(self.par[key]))
            if len(self.pdoc) > 0:
                newkey.setToolTip(self.pdoc[key])
                newitem.setToolTip(self.pdoc[key])
            newtype = QTableWidgetItem(type(self.par[key]).__name__)
            newtype.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row,0,newkey)
            self.table.setItem(row,1,newitem)
            self.table.setItem(row,2,newtype)

        self.tab2.layout.addWidget(self.table)
        self.tab2.setLayout(self.tab2.layout)
        
        
        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        

class MainWindow(QMainWindow):
    def __init__(self,par,pdoc={}):
        super().__init__()
        self.setup(par,pdoc)
        
    
    def setup(self,par,pdoc):
        self.setWindowTitle('pySFDI: an Uber SFDI GUI')
        self.setGeometry(0,50,800,480)
        self.tabWidget = tabLayout(self,par,pdoc)
        
        self.setCentralWidget(self.tabWidget)
        
        self.show()