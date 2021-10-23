import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()


while True:
    ret, image = cap.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    _,threshold = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
#    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
#    for cnt in contours:
#        (x, y, w, h) = cv2.boundingRect(cnt)
#        cv2.drawContours(image, [cnt], -1, (0,0,255), 3)
#        cv2.rectangle(image , (x,y),(x + w, y + h), (255, 0, 255), 2)
#        break

    faces = detector(gray_image)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)

#    cv2.imshow("Threshold", threshold)
#    cv2.imshow("Gray Frame", gray_image)
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
