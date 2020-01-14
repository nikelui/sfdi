# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:12:24 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys,json
from collections import OrderedDict
from PyQt5.QtWidgets import (QMainWindow,QTabWidget,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,
                QTableWidget,QTableWidgetItem,QHeaderView,QLabel,QStyle,QSpacerItem,QSizePolicy,
                QDesktopWidget,QSlider)
from PyQt5.QtCore import Qt,QThread,pyqtSignal,QSize
from PyQt5.QtGui import (QPixmap,QImage,QFont)

sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import sfdi
import PyCapture2 as pc
import numpy as np

class Thread(QThread):
    """Need to use thread, or it will freeze main window execution"""
    changePixmap = pyqtSignal(QImage)
    play = False
    
    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)       
        self.cameraSettings(parent)

    def cameraSettings(self,parent):
        self.cam = sfdi.setCamera_pg(num=0,res=parent.par['res'],fps=parent.par['fps']) # Set-up Camera
        #try:
        #    self.cam.startCapture()
        #except pc.Fc2error as fc2Err:
        #    print('Error starting capture: %s' % fc2Err)
    
    def run(self):
        try:
            self.cam.startCapture()
        except pc.Fc2error as fc2Err:
            print('Error starting capture: %s' % fc2Err)
        while True:
            try:
                im = self.cam.retrieveBuffer()
                im = im.convert(pc.PIXEL_FORMAT.BGR) # from RAW to color (BGR 8bit)
                a = np.array(im.getData())
                rawImage = QImage(a.data, im.getCols(), im.getRows(), QImage.Format_RGB888)
                #frame = sfdi.camCapt_pg(self.cam)
                #rawImage = QImage(frame.data,frame.shape[1],frame.shape[0],QImage.Format_RGB888)
                self.changePixmap.emit(rawImage.rgbSwapped())
            except pc.Fc2error as fc2Err:
                print('Error retrieving buffer: %s' % fc2Err)
        self.cam.stopCapture()
        self.cam.disconnect()


class tabLayout(QWidget):
    
    def setTab1(self,parent):
        """Create the first tab to show video feed and controls."""
        self.tab1 = QWidget()
        self.tab1.layout = QHBoxLayout(self) # Outer layout        
        self.tab1.layout.setContentsMargins(0,0,0,0)
        self.tab1.layout.setSpacing(0)
        
        ## Video goes here
        ## TODO: show a pixmap in the label, obtained from real-time camera feed
        self.pix = QLabel(self)
        self.pix.setObjectName('pix')
        self.pix.setFixedSize(640,480) # Maybe this is not the best solution
        
        ##TODO: add a scrollbar and two buttons to control exposure, on top of video
        hlayout = QHBoxLayout() # Layout for exposure bar and buttons
        
        ## Get info about exposure time
        info = parent.cam.getPropertyInfo(pc.PROPERTY_TYPE.SHUTTER)
		
        ## Set a limit to exposure to 500ms (to use when you disable fps)
        self.expMin = info.absMin
        if info.absMax > 500:
            self.expMax = 500
            self.tstep = 0.5
        else:
            self.expMax = info.absMax
            self.tstep = (info.absMax - info.absMin)/542
        self.expbar = QSlider(Qt.Horizontal)
        
        # Since the scrollbar only takes int, need to make some calculations
        self.expbar.setMaximum(543)
        self.expbar.setMinimum(1)
        self.expbar.setSingleStep(1)
        self.expbar.setTickPosition(QSlider.NoTicks)
        self.expbar.valueChanged.connect(lambda: self.expChange(parent))

        ## DEBUG
        print("%f, %f, %f" % (self.expMin,self.expMax,self.tstep))
        
        #hlayout.addWidget(self.expbar)
        
        ## Controls go here
        vlayout = QVBoxLayout(self) # Vertical layout for controls
        vlayout.setContentsMargins(0,0,0,0)
        vlayout.setSpacing(10)
        vlayout.setAlignment(Qt.AlignLeft)
        
        vSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
                
        hist = QLabel(self)
        hist.setObjectName('hist')
        hist.setFixedSize(160,100)
        
        vlayout.addWidget(hist)
        
        play = QPushButton()
        play.setObjectName('play')
        play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        play.setIconSize(QSize(40,40))
        play.setMinimumHeight(60)
        vlayout.addWidget(play)
        
        stop = QPushButton()
        stop.setObjectName('stop')
        stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        stop.setIconSize(QSize(40,40))
        stop.setMinimumHeight(60)
        vlayout.addWidget(stop)
        
        
        start = QPushButton("Start\nAcquisition")
        start.setObjectName('start')
        start.setFont(QFont('Serif',20))
        start.setMinimumHeight(50)
        vlayout.addWidget(start)
        vlayout.addSpacerItem(vSpacer)
        
        ## Label to show exposure time
        ltitle = QLabel()
        ltitle.setObjectName('ltitle')
        ltitle.setText('Exp. time')
        
        self.expLabel = QLabel() # This needs to be changed dynamically
        self.expLabel.setObjectName('expLabel')
        self.expLabel.setText('%.2fms' % (self.expbar.value() * self.tstep))
        
        vlayout.addWidget(ltitle)
        vlayout.addWidget(self.expLabel)
        
        # Test
        vlayout.addWidget(self.expbar)
        
        self.tab1.layout.addWidget(self.pix,alignment=Qt.AlignTop)
        self.tab1.layout.addLayout(vlayout)
        self.tab1.setLayout(self.tab1.layout)
        
        th = Thread(parent)
        th.changePixmap.connect(lambda p: self.setPixMap(p))
        th.start()
        
    def setTab2(self,parent):
        """Create second tab, to display and modify the parameters"""
        self.tab2 = QWidget()
        # Create second tab [parameters]
        self.tab2.layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(3)
        self.table.setRowCount(len(parent.par))
        self.table.setHorizontalHeaderLabels(('name','value','type'))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(False)
        # Populate table
        for row,key in enumerate(parent.par.keys()):
            newkey = QTableWidgetItem(key)
            newkey.setFlags(Qt.ItemIsEnabled)
            newitem = QTableWidgetItem(str(parent.par[key]))
            if len(parent.pdoc) > 0:
                newkey.setToolTip(parent.pdoc[key])
                newitem.setToolTip(parent.pdoc[key])
            newtype = QTableWidgetItem(type(parent.par[key]).__name__)
            newtype.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row,0,newkey)
            self.table.setItem(row,1,newitem)
            self.table.setItem(row,2,newtype)

        self.tab2.layout.addWidget(self.table)
        self.tab2.setLayout(self.tab2.layout)
    
    def setPixMap(self, p):
        """Slot to update pixmap from signal emitted by camera during acquisition"""
        p = QPixmap.fromImage(p)    
        p = p.scaled(640, 480, Qt.KeepAspectRatio)
        self.pix.setPixmap(p)
    
    def expChange(self,parent):
        """Slot to update exposure time"""
        e = self.expbar.value() # This is an int
        e = float(e * self.tstep) # convert to the proper value
        #print("Exposure time: %.1fms" % e)
        self.expLabel.setText("%.2fms" % e)
        try:
            parent.cam.setProperty(type=pc.PROPERTY_TYPE.SHUTTER,absValue=e)
        except pc.Fc2error as fc2Err:
            print('Error setting exposure: %s' % fc2Err)
        
    
    def __init__(self,parent):
        super(QWidget,self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        # Add tabs
        self.tabs = QTabWidget()
        self.setTab1(parent)
        self.setTab2(parent)
        self.tabs.addTab(self.tab1,"Capture")
        self.tabs.addTab(self.tab2,"Parameters")
        
        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loadParams()
        self.setup()
    
    def loadParams(self):
        f = open('parameters.json','r')
        self.par = json.load(f,object_pairs_hook=OrderedDict)
        f.close()
        
        f = open('documentation.json','r')
        self.pdoc = json.load(f,object_pairs_hook=OrderedDict)
        f.close()
        
        # Automatically detect screen resolution on startup
        screen1 = QDesktopWidget().screenGeometry(0) # main screen
        screen2 = QDesktopWidget().screenGeometry(1) # secondary screen / projector
        self.par['W'] = int(screen1.width())
        self.par['H'] = int(screen1.height())
        self.par['xRes'] = int(screen2.width())
        self.par['yRes'] = int(screen2.height())
        
    
    def setup(self):
        self.setWindowTitle('pySFDI: an Uber SFDI GUI')
        self.setGeometry(0,50,800,480) # TODO: smarter resizing?
        self.cam = sfdi.setCamera_pg(num=0,res=self.par['res'],fps=self.par['fps']) # Set-up Camera
        self.tabWidget = tabLayout(self)
        self.setStyleSheet(open("../resources/base/stylesheet.qss").read())
        
        self.setCentralWidget(self.tabWidget)
        
        self.show()