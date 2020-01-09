import sys
from PyQt5 import QtWidgets

def main():
    sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
    print(str(sizeObject.height()) +' x '+ str(sizeObject.width()))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main()
    sys.exit(app.exec_())