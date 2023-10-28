# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
# @Author  : Lei
# @Email   : 6222ppt@gmail.com
# @File    : window.py
# @Software: PyCharm
# @Brief   : Graphical interface" or "Graphical user interface (GUI)

import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil


class MainWindow(QTabWidget):
    # Initialize
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('Identification system')  # todo Modifying the system name
        # Model initialization
        self.model = tf.keras.models.load_model("models/mobilenet_fv.h5")  # todo Modifying the model name
        self.to_predict_name = "images/tim9.jpeg"  # todo Modify the initial image, which will be placed in the images directory
        self.class_names = ['Potato', 'Cherry Tomatoes', 'Cabbage', 'Green Chinese Onion', 'Pear', 'Carrots', 'Mango', 'Apple', 'Tomatoes', 'Leeks', 'Banana', 'Cucumber']  # todo 修改类名，这个数组在模型训练的开始会输出
        self.resize(900, 700)
        self.initUI()

    # Initialize the interface and set the interface layout
    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('Calibri', 15)

        # The main page, which sets components and places them on the layout
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("Sample")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" Upload pictures ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" Start recognition ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        label_result = QLabel(' Name ')
        self.result = QLabel("Wait recognition")
        label_result.setFont(QFont('Calibri', 16))
        self.result.setFont(QFont('Calibri', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        # Regarding the page, set up the components and place the components on the layout
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('Welcome to the intelligent identification system')  # todo Modify welcome words
        about_title.setFont(QFont('Calibri', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/bj.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel("Author:Lei")  # todo Change author information
        label_super.setFont(QFont('Calibri', 12))
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        # Add comment
        self.addTab(main_widget, 'Main')
        self.addTab(about_widget, 'About')
        self.setTabIcon(0, QIcon('images/Main.png'))
        self.setTabIcon(1, QIcon('images/About.png'))

    # Upload and display pictures
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # Open the file selection box to select the file
        img_name = openfile_name[0]  # Get picture name
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]  # Move the picture to the current directory
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)  # Open picture
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)  # Adjust the size of the picture to a height of 400, which is convenient for the interface display
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))  # Resize the picture to 224*224 for model inference
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))
            self.result.setText("Wait recognition")

    # Prediction picture
    def predict_img(self):
        img = Image.open('images/target.png')  # Read picture
        img = np.asarray(img)  # Convert the image to an array of numpy
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))  # Input images into the model to get results
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]  # Get the corresponding name
        self.result.setText(result)  # Display on the interface

    # Screen close event, asking the user whether to close
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'Quit',
                                     "Do you want to exit the program?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
