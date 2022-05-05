import sys
from Ui_template_ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QPushButton, QPlainTextEdit,QLabel,QMessageBox
import os
import pandas as pd
import pyqtgraph as pg
import matplotlib.pyplot as plt
import numpy as np
import tkinter
import tkinter.filedialog  # 该包用于通过文件对话窗口，选择本地的某个文件，获取该文件的路径
import tensorflow as tf
from tensorflow import keras
import configparser as cfg
class myclass():
    def __init__(self) -> None:
        pass
class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)
        #在此输入connect链接
        self.show()
        self.path = os.getcwd()# 获取当前目录，os模块函数
        self.cfg=cfg.ConfigParser()

        self.label_curfile.setText(self.path)# 更改lineEdit控件文字内容
        self.label_3.setText(self.path)# 更改lineEdit控件文字内容
        self.openexcel.clicked.connect(self.openfile)
        self.weight_button.clicked.connect(self.loadweights)
        self.predict_button.clicked.connect(self.predict)
        self.training.clicked.connect(self.train)
        self.plot = self.curPlt(self.display, "x", "y", "数据可视化", True)


        if(os.path.exists('config.ini')):# 判断配置文件是否存在
            self.cfg.read('config.ini')
            self.filename=self.cfg.get("filename","filename")
            data = pd.read_excel(self.filename) #reading file
            height,width = data.shape
            x = np.zeros((height,width),dtype=float)
            for i in range(0,height):
                for j in range(0,width):
                    x[i][j] = data.iloc[i,j]
            self.xlist=x[:,0]
            self.ylist=x[:,1]
            self.scale=np.max(self.ylist)
            self.scaleylist=self.ylist/self.scale
            self.scaletxt.setText(str(self.scale))
            self.plot.setData(self.xlist,self.ylist)
        else:
            self.cfg.add_section("filename")
            self.cfg.add_section("scale")



        self.model = keras.Sequential([
            keras.layers.Dense(64,activation=tf.nn.relu, input_shape=(1,)),
            keras.layers.Dense(64,activation=tf.nn.relu),
            keras.layers.Dense(128,activation=tf.nn.relu),
            keras.layers.Dense(256,activation=tf.nn.relu),
            keras.layers.Dense(256,activation=tf.nn.relu),
            keras.layers.Dense(128,activation=tf.nn.relu),
            keras.layers.Dense(64,activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

    def curPlt(self,widget,xLabel,yLabel,title,isPre):
        # 曲线图画布初始化

        verticalLayout = QtWidgets.QVBoxLayout(widget)
        win = pg.GraphicsLayoutWidget(widget)
        win.setBackground((255, 255, 255))  # 背景色)
        verticalLayout.addWidget(win)
        if isPre:
            p = win.addPlot(title="<span style='font-size:12px;color:black'>" + title + "</span>")
            p.showGrid(x=True, y=True)
        else:
            p = win.addPlot(title="<span style='font-size:16px;color:black'>" + title + "</span>")
            p.showGrid(x=True, y=True)
            p.setLabel(axis="left", text="<span style='font-size:16px;color:black;font-family: Arial'>"+yLabel+"</span>")
            p.setLabel(axis="bottom", text="<span style='font-size:16px;color:black;font-family: Arial'>"+xLabel+"</span>")
        myPen = pg.mkPen({'color': (0, 134, 139), 'width': 2})
        curve=p.plot(pen=myPen, name="y1")
        font = QtGui.QFont()
        font.setPointSize(12)
        axisPen = pg.mkPen({'color': (0, 0, 0), 'width': 1})
        left_axis = p.getAxis("left")
        left_axis.enableAutoSIPrefix(False)
        left_axis.setStyle(tickFont=font)
        left_axis.setPen(axisPen)
        bot_axis = p.getAxis("bottom")
        bot_axis.enableAutoSIPrefix(False)
        bot_axis.setStyle(tickFont=font)
        bot_axis.setPen(axisPen)
        return curve
    def errorShow(self,info):
        # 弹出错误框，参数info为弹框所要显示的文字
        QMessageBox.critical(self,'错误',info ,QMessageBox.Yes)
    def openfile(self):
        root = tkinter.Tk()
        root.withdraw()
        filename = tkinter.filedialog.askopenfilename(parent=root, initialdir=self.path, title='pick a trace')
        if filename.strip():
            self.excelfile=filename
            self.label_curfile.setText(filename)
            self.cfg.set('filename','filename',filename)
            data = pd.read_excel(filename) #reading file
            height,width = data.shape
            x = np.zeros((height,width),dtype=float)
            for i in range(0,height):
                for j in range(0,width):
                    x[i][j] = data.iloc[i,j]
            self.xlist=x[:,0]
            self.ylist=x[:,1]
            self.scale=np.max(self.ylist)
            self.cfg.set('scale','scale',str(self.scale))
            with open('config.ini','w+') as f:# 创建配置文件
                self.cfg.write(f)
            self.scaleylist=self.ylist/self.scale
            print('scale='+str(self.scale))
            self.shape=self.xlist.shape
        self.scaletxt.setText(str(self.scale))
        self.plot.setData(self.xlist,self.ylist)
    def train(self):
        times=int(self.times.text())
        history=self.model.fit(self.xlist, self.scaleylist, epochs=times)
        print(history.history.keys())
        self.model.save_weights('./weigths.h5')
        '''
        plt.plot(history.history['loss'])
        plt.plot(history.history['accuracy'])
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Loss', 'accuracy'], loc='upper right')
        plt.show()'''
        predictions = self.model.predict(self.xlist)
        plt.title('Traing data predicted vs actual values')
        plt.plot(self.xlist,self.ylist,'b.', label='Actual')
        plt.plot(self.xlist,predictions*self.scale,'r.', label='Predicted')
        plt.legend()
        plt.show()
    def loadweights(self):
        self.model.load_weights('./weigths.h5')
        self.model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        QMessageBox.information(self,'提示','模型参数加载成功' ,QMessageBox.Yes)
        print('load weight finish')
    def predict(self):
        x=np.array([float(self.input.text())])
        y=self.model.predict(x)
        self.output.setText(str(y*self.scale)[1:-2])



if __name__ == "__main__":  # 主函数执行
    app = QApplication(sys.argv)
    globFont = QtGui.QFont()
    globFont.setFamily('Microsoft YaHei')
    globFont.setPointSize(10)
    app.setFont(globFont)
    MainUI = MainWindow()  # 将主界面定义为欢迎界面，程序运行至此处开始调用MainWindow()类
    sys.exit(app.exec_())  # 程序执行完毕后关闭
