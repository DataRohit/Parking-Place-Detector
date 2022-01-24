import cv2
import pickle
import cvzone
import numpy as np

# Video Feed
cap = cv2.VideoCapture("carPark.mp4")

width, height = 104, 45

with open("CarParkPos", "rb") as f:
    pos_list = pickle.load(f)

def check_parking_space(imgPro):
    for pos in pos_list:
        x, y = pos

        img_crop = imgPro[y : y + height, x : x + width]

        count = cv2.countNonZero(img_crop)

        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1.2, thickness=2, offset=0)

        if count < 700:
            color = (0, 255, 0)
            thickness = 5
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, thickness)

while True:
    # Loop Over the Video
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedianBlur = cv2.medianBlur(imgThreshold, 5)
    imgDilate = cv2.dilate(imgMedianBlur, np.ones((3, 3), np.uint8), iterations=1)

    check_parking_space(imgDilate)

    cv2.imshow("Video", img)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break