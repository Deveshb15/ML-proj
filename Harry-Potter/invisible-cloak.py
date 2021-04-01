import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():

    ret, frame = cap.read()

    if ret:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv",hsv)
        # blue = np.uint8([[[0, 0, 255]]])
        # hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        
        blue_lower = np.array([100,150,0])
        blue_upper = np.array([140,255,255])

        mask = cv2.inRange(hsv, blue_lower, blue_upper)
        # cv2.imshow("mask", mask)

        # ignoring blue
        part1 = cv2.bitwise_and(back, back, mask=mask)
        # cv2.imshow("part1", part1)

        mask = cv2.bitwise_not(mask)

        # put black on blue
        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow("part2", part2)
        
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(part1 + part2, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('cloak', closing)

        if cv2.waitKey(5) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()