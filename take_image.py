
import cv2
import time

capture = cv2.VideoCapture(0)
print("set focus...")
time.sleep(5)
for i in range(10):
    ret, frame = capture.read()
    filename = f'calibration_images/image{i}.png'
    cv2.imwrite(filename, frame)
    print("save " + filename)
    time.sleep(4)
