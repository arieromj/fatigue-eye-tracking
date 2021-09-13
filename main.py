import cv2
import numpy as np
import dlib
from math import hypot


cap = cv2.VideoCapture(1)                                                   #class for video capturing from my webcam.
detector = dlib.get_frontal_face_detector()                                 #returns the default face detector.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   #takes in an image region containing an eye and outputs a set of point locations.

#Function that defines the midpoint between 2 points
def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def blinking(eye_points, facial_landmarks):
    x = facial_landmarks.part(eye_points[0]).x                              # x value from point 36 of "shape_predictor_68_face_landmarks.dat"
    y = facial_landmarks.part(eye_points[0]).y                              # y Value from point 36 of "shape_predictor_68_face_landmarks.dat"
    x1 = facial_landmarks.part(eye_points[1]).x                             # x Value from point 39 of "shape_predictor_68_face_landmarks.dat"
    y1 = facial_landmarks.part(eye_points[1]).y                             # y Value from point 39 of "shape_predictor_68_face_landmarks.dat"

    # Horizontal line between points 36 and 39 from "shape_predictor_68_face_landmarks.dat"
    left_point = (x, y)
    right_point = (x1, y1)
    # Length of horizontal line
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    # Vertical line between middle points of 37,38 and 40,41 points from "shape_predictor_68_face_landmarks.dat"
    center_top = midpoint(landmarks.part(37), landmarks.part(38))           # middle points of 37 and 38 from "shape_predictor_68_face_landmarks.dat"
    center_bottom = midpoint(landmarks.part(41), landmarks.part(40))        # middle points of 40 and 41 from "shape_predictor_68_face_landmarks.dat"
    # Length of vertical line
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    # Ratio between horizontal line and vertical line. Useful to detect if the eye is closed or blinking
    ratio = hor_line_length / ver_line_length
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    #Isolating the eye to detect gaze direction
    height, width, _ = image.shape
    mask = np.zeros((height, width), np.uint8)                              #Black window with the same size of 'image'
    cv2.polylines(mask, [eye_region], True, 255, 2)                         #draw a polygon around eye region inside of black window mask
    cv2.fillPoly(mask, [eye_region], 255)                                   #fill the polygon inside of black window mask
    eye = cv2.bitwise_and(gray_image, gray_image, mask=mask)                #Insert the eye of gray_image in the polygon region of mask

    min_x = np.min(eye_region[:, 0])                                        #minimum x value inside of left eye region
    max_x = np.max(eye_region[:, 0])                                        #maximum x value inside of left eye region
    min_y = np.min(eye_region[:, 1])                                        #minimum y value inside of left eye region
    max_y = np.max(eye_region[:, 1])                                        #maximum y value inside of left eye region

    #Count how many white pixels are in the left and right side of the eye in order to detect the gaze direction
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if right_side_white == 0:
        gaze_ratio = 0.7
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


while True:
    ret, image = cap.read()                                                 #wait for a new frame from camera and store it into 'image'. 'ret' will obtain return value from getting the camera frame, either true or false.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                    #gray image can save computation resources.
    _,threshold = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

    faces = detector(gray_image)        #'faces' is a array where we have all the faces.
    for face in faces:
        w, z = face.left(), face.top()
        w1, z1 = face.right(), face.bottom()
        cv2.rectangle(image, (w, z), (w1, z1), (255, 0, 255), 2)            #Draw rectangles in faces.

        landmarks = predictor(gray_image, face)                             #Object Detection -> Faces

        #Blink Detection
        left_eye_blink = blinking([36,39], landmarks)
        right_eye_blink = blinking([36, 39], landmarks)
        eye_blink = (left_eye_blink + right_eye_blink)/2

        if eye_blink > 4:
            cv2.putText(image, "OLHO FECHADO", (0, 50), font, 2, (255, 0, 0))

        #Gaze Detection
        gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        #cv2.putText(image, str(gaze_ratio), (50, 400), font, 2, (0, 0, 255), 3)

        if gaze_ratio < 0.5:
            cv2.putText(image, "ESQUERDA", (50,100), font, 2, (0,0,255), 3)
        elif 0.5 < gaze_ratio < 1.1:
            cv2.putText(image, "CENTRO", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(image, "DIREITA", (50, 100), font, 2, (0, 0, 255), 3)

    cv2.imshow("Frame", image)                                              #Displays an image in the specified window.
    key = cv2.waitKey(1)                                                    #Wait key in periods of 1ms.
    if key == 27:                                                           #Break the loop when 'ESC' is pressed.
        break

cap.release()
cv2.destroyAllWindows()