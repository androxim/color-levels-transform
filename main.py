import sys
from mywindow import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

import cv2
import numpy as np
from PIL import Image
from NeuroPy import NeuroPy

"""
	TO DO:
	1. add raw brain signal processing and estimation of frequencies / attention through FFT
	2. add plots of brain waves
"""

class MyWin(QtWidgets.QMainWindow):
    started = False
    ncolors = 256
    colors_by_att = False
    MindWave_obj = NeuroPy("COM3")
    cam = cv2.VideoCapture()
    pwidth, pheight = 0, 0
    pil_img = Image.init()
    pixmap = QtGui.QPixmap

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Color levels | PyQt | OpenCV | NeuroSky MindWave")
        self.setGeometry(400, 100, 1241, 820)
        self.ui.pushButton_3.clicked.connect(self.Ext)
        self.ui.pushButton_4.clicked.connect(self.loadImage)
        self.ui.pushButton_5.clicked.connect(self.saveImage)
        self.ui.horizontalSlider.setValue(self.ncolors)
        self.ui.horizontalSlider.valueChanged.connect(self.colorsChanged)
        self.ui.checkBox.stateChanged.connect(self.colorsModeChanged)
        self.ui.checkBox_2.stateChanged.connect(self.cameraModeChanged)
        self.pwidth = self.ui.label.width()
        self.pheight = self.ui.label.height()
        self.pil_img = Image.open("eye.jpg")
        self.pixmap = QtGui.QPixmap.fromImage(self.pilImgToQImg_resized(self.pil_img))
        self.ui.label.setPixmap(self.pixmap)
        if not self.MindWave_obj.threadRun:
            try:
                self.MindWave_obj.start()
                self.statusBar().showMessage(" === MindWave connected! === ")
            except:
                self.statusBar().showMessage(" === Connection to MindWave failed! === ")

    def pilImgToQImg_resized(self, pilimg):
        """	convert Pillow Image to QImage and resize choosen image to window area """
        ocvimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        dim = (self.pwidth, self.pheight)
        resized = cv2.resize(ocvimg, dim, interpolation=cv2.INTER_AREA)
        self.pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        return QtGui.QImage(resized, resized.shape[1], resized.shape[0], resized.shape[1] * 3,
                             QtGui.QImage.Format_RGB888).rgbSwapped()

    def saveImage(self):
        """ save current transformation of image to file """
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveeFileName()", "",
                                                  "Image Files (*.jpg *.png *.bmp)")
        if fileName:
            self.pixmap.save(fileName)

    def loadImage(self):
        """ load image from file """
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Image Files (*.jpg *.png *.bmp)")
        if fileName:
            self.pil_img = Image.open(fileName)
            self.pixmap = QtGui.QPixmap.fromImage(self.pilImgToQImg_resized(self.pil_img))
            self.ui.label.setPixmap(self.pixmap)

    def cameraModeChanged(self):
        """	turn on / off camera input """
        if not self.cam.isOpened():
            self.cam.open(0)
            self.started = True
        else:
            self.started = False
            self.cam.release()

    def colorsModeChanged(self):
        """ turn on / off color levels dependency on attention level (obtained from EEG device) """
        self.colors_by_att = not self.colors_by_att

    def colorsChanged(self):
        """ processing change of color levels """
        self.ncolors = self.ui.horizontalSlider.value()
        self.ui.progressBar.setValue(self.MindWave_obj.attention)
        self.ui.label_3.setText("COLOR LEVELS: " + str(self.ncolors))
        if not self.started:
            t_pil_img = self.pil_img.convert('P', palette=Image.ADAPTIVE, colors=self.ncolors).convert('RGB')
            im = t_pil_img.convert("RGBA")
            data = im.tobytes("raw", "RGBA")
            qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_RGBA8888)
            self.pixmap = QtGui.QPixmap.fromImage(qim)
            self.ui.label.setPixmap(self.pixmap)

    def update_flow(self):
        """ update attention value and processing camera input for transformation """
        self.ui.progressBar.setValue(self.MindWave_obj.attention)
        if self.started:
            ret, frame = self.cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_pil_img = Image.fromarray(frame)
            t_pil_img = t_pil_img.convert('P', palette=Image.ADAPTIVE, colors=self.ncolors).convert('RGB')
            im = t_pil_img.convert("RGBA")
            data = im.tobytes("raw", "RGBA")
            qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_RGBA8888)
            self.pixmap = QtGui.QPixmap.fromImage(qim).scaled(self.pwidth,self.pheight, QtCore.Qt.IgnoreAspectRatio)
            self.ui.label.setPixmap(self.pixmap)

        if self.MindWave_obj.attention > 0: 
            if self.colors_by_att:
                self.ncolors = self.MindWave_obj.attention 
                self.ui.horizontalSlider.setValue(self.ncolors)

    def Ext(self):
        """ exit the application """
        sys.exit()


if __name__== "__main__":
    img = Image.open('eye.jpg')
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(myapp.update_flow)
    timer.start(5)

    sys.exit(app.exec_())
