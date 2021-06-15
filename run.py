import sys
from UI.main import Ui_MainWindow
from vgg_classify import Classify
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import *


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.initialize()

    def initialize(self):
        self.setWindowTitle("图像分类软件")
        self.img_path = ""
        self.label = ""
        self.precision = 0.0
        self.label_showImage.setPixmap(QPixmap("bg.jpg"))

        # button slot
        self.pushButton_openImage.clicked.connect(lambda: self.btn_openImage())
        self.pushButton_classify.clicked.connect(lambda: self.btn_classify())
        self.menuExit.triggered.connect(lambda: self.btn_exit())

        # load model
        self.model = Classify("checkpoint/model.pth", 128, 3, True)
        self.textBrowser.append("Initialization is complete!\n")
        print("Initialization is complete")

    def btn_openImage(self):
        print("openImage")
        self.img_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', '图片文件 (*.jpg; *.png)')
        print(self.img_path)
        self.textBrowser.append("Load Image： " + str(self.img_path))
        self.label_openedImage.setText(str(self.img_path))
        self.label_showImage.setPixmap(QPixmap(self.img_path))

    def btn_exit(self):
        sys.exit()


    def btn_classify(self):
        if self.img_path == "":
            print("未导入图片")
            self.textBrowser.append("未导入图片")
        else:
            self.precision, self.label = self.model.classify(self.img_path)
            self.label_class.setText('label: ' + str(self.label))
            self.textBrowser.append('Label: ' + str(self.label))
            self.textBrowser.append('Rrecision: ' + str(self.precision))
            self.textBrowser.append('\n')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
