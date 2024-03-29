# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

# GUI Imports
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap
# LSB Imports
import time
import tensorflow as tf
import holisticModule as hm
import prepareData as prep
import numpy as np
import cv2
import sys
import pyttsx3

CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
            'Necesito', 'Quien Occidente', 'Saludos']


def play(word):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice", voices[2].id)
    engine.say(word)
    engine.runAndWait()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_gesture_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.source = 1  # CheckFlag if with cable
        self.model = 'C:\\Users\\DougC\\Desktop\\Special\\RNN_Final-151-0.75.model'
        self.modelType = "LSTM Stacked"
        self.draw = False
        self.sound = False
        self._run_flag = True

    def set_source(self,feed,modelType,path_model,draw,sound):
        self.source = feed
        self.modelType = modelType
        self.model = path_model
        self.draw = draw
        self.sound = sound

    def run(self):
        # Init objects
        detector = hm.HolisticDetector()
        prepare = prep.Preprocessor()
        new_model = tf.keras.models.load_model(self.model)
        # capture from web cam
        cap = cv2.VideoCapture(self.source)
        vec = np.zeros((84, 1))
        frame_counter = 0
        pastTime = 0
        print(self.modelType)
        if self.modelType == "LSTM Stacked":
            seq_length = 136
        if self.modelType == "LSTM BiDirectional":
            seq_length = 90
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                while vec.shape[1] < seq_length:
                    vec = np.column_stack((vec, np.zeros(84)))
                vec = np.asarray(vec, dtype=np.float32).reshape(-1, seq_length, 84)
                prediction = new_model.predict(vec, steps=1, verbose=0)
                inference = CATEGORY[np.argmax(prediction)]
                self.change_gesture_signal.emit(inference)
                if self.sound:
                    play(inference)
                break
            detector.find_pose(frame)
            if self.draw:
                detector.draw_pose(frame)
            detector.get_lm()
            mat1, mat2, mat3 = detector.get_matrix()
            feat_vec = prepare.get_distMat(mat1, mat2, mat3)
            feat_vec = np.array(feat_vec)
            mag = np.true_divide(feat_vec[:, 0], 2202)  # normalize dist for 1920x1080
            ang = np.true_divide(feat_vec[:, 1], 360)  # normalize ang for 360
            aux = np.concatenate((mag, ang))
            frame_counter += 1
            vec = np.column_stack((vec, aux))
            # new_mat = np.delete(vec, -1, axis=1)
            # new_mat = np.insert(new_mat, 0, aux, axis=1)
            # temp = np.asarray(new_mat, dtype= np.float32).reshape(-1, 136, 84)
            currentTime = time.time()
            fps = int(1 / (currentTime - pastTime))
            pastTime = currentTime
            cv2.putText(frame, f'FPS:{fps}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(frame, f'COUNT:{frame_counter}', (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            self.change_pixmap_signal.emit(frame)
            cv2.waitKey(1)
            if frame_counter < seq_length:
                continue
            else:
                frame_counter = 0
            vec = np.delete(vec, -1, axis=1)
            vec = np.asarray(vec, dtype=np.float32).reshape(-1, seq_length, 84)
            prediction = new_model.predict(vec, steps=1, verbose=0)
            inference = CATEGORY[np.argmax(prediction)]
            #print(inference)
            self.change_gesture_signal.emit(inference)
            if self.sound:
                play(inference)
            vec = np.zeros((84, 1))
        # shut down capture system
        cap.release()

    def resume(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = True
        self.wait()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 820)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lblVideoCap = QtWidgets.QLabel(self.centralwidget)
        self.lblVideoCap.setGeometry(QtCore.QRect(330, 20, 140, 21))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(14)
        self.lblVideoCap.setFont(font)
        self.lblVideoCap.setAlignment(QtCore.Qt.AlignCenter)
        self.lblVideoCap.setObjectName("lblVideoCap")
        self.lblDisplay = QtWidgets.QLabel(self.centralwidget)
        self.lblDisplay.setGeometry(QtCore.QRect(80, 70, 640, 360))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        self.lblDisplay.setFont(font)
        self.lblDisplay.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.lblDisplay.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lblDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.lblDisplay.setObjectName("lblDisplay")

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_gesture_signal.connect(self.update_text)
        # start the thread
        self.thread.start()

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(50, 520, 300, 120))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.radioCamera = QtWidgets.QRadioButton(self.groupBox)
        self.radioCamera.setGeometry(QtCore.QRect(60, 35, 120, 25))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        self.radioCamera.setFont(font)
        self.radioCamera.setObjectName("radioCamera")
        self.radioCamera.setChecked(True)
        self.radioVideo = QtWidgets.QRadioButton(self.groupBox)
        self.radioVideo.setGeometry(QtCore.QRect(60, 80, 120, 25))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        self.radioVideo.setFont(font)
        self.radioVideo.setObjectName("radioVideo")
        self.checkDraw = QtWidgets.QCheckBox(self.centralwidget)
        self.checkDraw.setGeometry(QtCore.QRect(160, 720, 70, 17))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        self.checkDraw.setFont(font)
        self.checkDraw.setObjectName("checkDraw")
        self.checkDraw.clicked.connect(lambda: self.get_draw())
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(410, 520, 300, 120))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.comboBoxModel = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBoxModel.setGeometry(QtCore.QRect(20, 50, 261, 22))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        self.comboBoxModel.setFont(font)
        self.comboBoxModel.setObjectName("comboBoxModel")
        self.comboBoxModel.addItem("")
        self.comboBoxModel.addItem("")
        # self.comboBoxModel.addItem("")
        # self.comboBoxModel.addItem("")

        self.bttnRun = QtWidgets.QPushButton(self.centralwidget)
        self.bttnRun.setGeometry(QtCore.QRect(490, 670, 140, 51))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        self.bttnRun.setFont(font)
        self.bttnRun.setObjectName("bttnRun")
        self.bttnRun.clicked.connect(lambda : self.proccess_info())
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setEnabled(True)
        self.layoutWidget.setGeometry(QtCore.QRect(160, 450, 481, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.layoutWidget.sizePolicy().hasHeightForWidth())
        self.layoutWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.layoutWidget.setFont(font)
        self.layoutWidget.setObjectName("layoutWidget")
        self.DisplayMenu = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.DisplayMenu.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.DisplayMenu.setContentsMargins(0, 0, 0, 0)
        self.DisplayMenu.setSpacing(6)
        self.DisplayMenu.setObjectName("DisplayMenu")
        self.bttnPlay = QtWidgets.QPushButton(self.layoutWidget)
        self.bttnPlay.clicked.connect(lambda: play(self.lblGesture.text().split(':')[1]))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bttnPlay.sizePolicy().hasHeightForWidth())
        self.bttnPlay.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.bttnPlay.setFont(font)
        self.bttnPlay.setObjectName("bttnPlay")
        self.DisplayMenu.addWidget(self.bttnPlay)
        self.lblGesture = QtWidgets.QLabel(self.layoutWidget)
        self.lblGesture.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblGesture.sizePolicy().hasHeightForWidth())
        self.lblGesture.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.lblGesture.setFont(font)
        self.lblGesture.setObjectName("lblGesture")
        self.DisplayMenu.addWidget(self.lblGesture)
        self.checkAudio = QtWidgets.QCheckBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkAudio.sizePolicy().hasHeightForWidth())
        self.checkAudio.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.checkAudio.setFont(font)
        self.checkAudio.setObjectName("checkAudio")
        self.DisplayMenu.addWidget(self.checkAudio, 0, QtCore.Qt.AlignHCenter)

        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(50, 690, 301, 25))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        self.layoutWidget1.setFont(font)
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.txtFile = QtWidgets.QLineEdit(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        self.txtFile.setFont(font)
        self.txtFile.setObjectName("txtFile")
        self.horizontalLayout.addWidget(self.txtFile)
        self.bttnFile = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        self.bttnFile.setFont(font)
        self.bttnFile.setObjectName("bttnFile")
        self.horizontalLayout.addWidget(self.bttnFile)
        self.bttnFile.clicked.connect(lambda: self.get_video())
        self.lblVideoPath = QtWidgets.QLabel(self.centralwidget)
        self.lblVideoPath.setGeometry(QtCore.QRect(50, 660, 181, 16))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblVideoPath.sizePolicy().hasHeightForWidth())
        self.lblVideoPath.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(10)
        self.lblVideoPath.setFont(font)
        self.lblVideoPath.setScaledContents(False)
        self.lblVideoPath.setObjectName("lblVideoPath")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lblVideoCap.setText(_translate("MainWindow", "Video Capture"))
        self.lblDisplay.setText(_translate("MainWindow", "Display"))
        self.groupBox.setTitle(_translate("MainWindow", "Select source"))
        self.radioCamera.setText(_translate("MainWindow", "Camera"))
        self.radioVideo.setText(_translate("MainWindow", "Video"))
        self.checkDraw.setText(_translate("MainWindow", "Draw"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Select model"))
        self.comboBoxModel.setItemText(0, _translate("MainWindow", "LSTM Stacked"))
        # self.comboBoxModel.setItemText(1, _translate("MainWindow", "LSTMCuDNN Stacked"))
        self.comboBoxModel.setItemText(1, _translate("MainWindow", "LSTM BiDirectional"))
        # self.comboBoxModel.setItemText(3, _translate("MainWindow", "LSTM BiDirectional Stacked"))
        self.bttnRun.setText(_translate("MainWindow", "RUN"))
        self.bttnPlay.setText(_translate("MainWindow", "Play!"))
        self.lblGesture.setText(_translate("MainWindow", "Gesture: "))
        self.checkAudio.setText(_translate("MainWindow", "Audio"))
        self.bttnFile.setText(_translate("MainWindow", "Select"))
        self.lblVideoPath.setText(_translate("MainWindow", "Select video path"))

    def get_video(self):
        file, _ = QFileDialog.getOpenFileName(None, 'Select video file',
                                              r'C:\\Users\\DougC\\Desktop\\LSBv1\\')
        self.txtFile.setText(file)

    def get_draw(self):
        flag = self.checkDraw.isChecked()
        return flag

    def get_audio(self):
        flag = self.checkAudio.isChecked()
        return flag

    def get_model(self):
        model = self.comboBoxModel.currentText()
        return model

    def get_source(self):
        flagVideo = self.radioVideo.isChecked()
        return flagVideo

    def get_audio(self):
        audio = self.checkAudio.isChecked()
        return audio

    def gen_report(self):
        flagSource = self.get_source()
        if not flagSource:
            source = 1  # CheckFlag if with cable
        else:
            source = self.txtFile.text()
        draw = self.get_draw()
        modelType = self.get_model()
        audio = self.get_audio()
        print(f"Report:\n"
              f"Source: {source}\n"
              f"Draw: {draw}\n"
              f"Model: {modelType}\n"
              f"Audio: {audio}")
        return source,draw,modelType,audio

    # @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.lblDisplay.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 360, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_text(self, pred):
        self.lblGesture.setText("Gesture:"+pred)

    def proccess_info(self):
        self.thread.stop()
        source,draw,modelType,audio = self.gen_report()
        model = self.get_model_path(modelType)
        self.thread.set_source(source,modelType,model,draw,audio)
        self.thread.resume()
        self.thread.start()

    def get_model_path(self,type):
        if type == "LSTM Stacked":
            return 'C:\\Users\\DougC\\Desktop\\Special\\RNN_Final-151-0.75.model'
        if type == "LSTM BiDirectional":
            return "C:\\Users\\DougC\\Desktop\\Special\\test80.h5"


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
