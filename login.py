import sys
import subprocess
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        self.setGeometry(300, 300, 300, 150)
        self.setStyleSheet("background-color: #F5F5F5;")
        
        self.username_label = QLabel("Username", self)
        self.username_label.setAlignment(Qt.AlignCenter)
        self.password_label = QLabel("Password", self)
        self.password_label.setAlignment(Qt.AlignCenter)
        
        self.username_input = QLineEdit(self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("Login", self)
        self.login_button.clicked.connect(self.login)

        layout = QVBoxLayout()
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)
        self.setLayout(layout)
    
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if username == "admin" and password == "admin":
            # Open code.py or any other script using subprocess
            subprocess.call(['python', 'main.py'])
            self.close()
        else:
            QtWidgets.QMessageBox.warning(self, "Login Failed", "Incorrect username or password")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())
