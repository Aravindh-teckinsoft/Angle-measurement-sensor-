import sys
import time
import gpiod
from gpiod.line import Direction, Value
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

LINE = 2

class LEDControl(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LED Control")

        self.button_on = QPushButton("ON", self)
        self.button_on.clicked.connect(self.turn_on_led)
        self.button_on.setGeometry(50, 50, 200, 50)

        self.button_off = QPushButton("OFF", self)
        self.button_off.clicked.connect(self.turn_off_led)
        self.button_off.setGeometry(50, 150, 200, 50)
        
     
        self.gpio_request = gpiod.request_lines(
            "/dev/gpiochip0",
            consumer="led-control",
            config={
                LINE: gpiod.LineSettings(
                    direction=Direction.OUTPUT, output_value=Value.INACTIVE
                )
            },
        )

    def turn_on_led(self):
        if self.gpio_request:
            self.gpio_request.set_value(LINE, Value.ACTIVE)

    def turn_off_led(self):
        if self.gpio_request:
            self.gpio_request.set_value(LINE, Value.INACTIVE)
            
    def close_window(self, event):
        if self.gpio_request:
            self.gpio_request.release()

 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LEDControl()
    window.setGeometry(100, 100, 300, 250)
    window.show()
    sys.exit(app.exec_())


