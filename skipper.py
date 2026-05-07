import pyautogui
import time
from ultralytics import YOLO
import webbrowser
import os

# load skipper face
path = os.path.abspath("./interface.html")
webbrowser.open(f"file://{path}")
time.sleep(10)

model = YOLO("best.pt")

running = True

while running:
    # watch screen
    pyautogui.screenshot("screen.png")
    results = model("screen.png")
    boxes = results[0].boxes.xyxy.numpy().astype(int)
    print(results[0].boxes)
    confidence = results[0].boxes.conf
    if confidence.numel() > 0:
        if confidence[0] >= 0.5:
            x1, y1, x2, y2 = boxes[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            pyautogui.moveTo(center_x, center_y, 1)
            pyautogui.click()
            time.sleep(1)
