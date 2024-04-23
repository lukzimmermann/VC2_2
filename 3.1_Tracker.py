import numpy as np
import cv2
import cv2.aruco as aruco

#create your marker using http://chev.me/arucogen/
markerssize = 0.1 #in m
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

# open calibration
cv_file = cv2.FileStorage("calib.yaml", cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

#------------------ ARUCO TRACKER ---------------------------

# get webcam stream
cap = cv2.VideoCapture(0)

foc_x = 1209.0
foc_y = 1201.9
principle_x = 960.9
principle_y = 530.2

mtx = np.array([[foc_x, 0, principle_x],
                [0, foc_y, principle_y],
                [0, 0, 1]], dtype=np.float64)
dist = np.array([-0.028, -0.0522, 0.00586, -0.00086, 1.616], dtype=np.float64)

cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)

    if np.all(ids != None):
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, markerssize, mtx, dist)

        length=0.1
        thickness=20

        for i in range(0, ids.size):
            axis = np.float32([[0,0,0],[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
            imgpts, jac= cv2.projectPoints(axis, rvec[i], tvec[i], mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1,2)

            

            #print("Marker "+str(ids[i][0])+" tvec: "+str(tvec[i])+" rvec: "+str(rvec[i]))

            aruco.drawDetectedMarkers(frame, corners)

            cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), thickness);
            cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), thickness);
            cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), thickness);

            cv2.imshow('frame',frame)

        if ids.size > 1:
            color = (255, 0, 255)

            x_distance = tvec[0][0][0] - tvec[1][0][0]
            y_distance = tvec[0][0][1] - tvec[1][0][1]
            z_distance = tvec[0][0][2] - tvec[1][0][2]
            distance = np.linalg.norm(tvec[1]-tvec[0])
            cv2.putText(frame, f"{str("Distance:"):<10} {distance:>5.2f}m", (20,128), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2,cv2.LINE_AA)
            cv2.putText(frame, f"{str("x:"):<10} {x_distance:>5.2f}m", (20,128+45), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2,cv2.LINE_AA)
            cv2.putText(frame, f"{str("y:"):<10} {y_distance:>5.2f}m", (20,128+90), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2,cv2.LINE_AA)
            cv2.putText(frame, f"{str("z:"):<10} {z_distance:>5.2f}m", (20,128+135), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2,cv2.LINE_AA)

            color = (0, 0, 0)
            imgpts1, jac= cv2.projectPoints(axis, rvec[0], tvec[0], mtx, dist)
            imgpts1 = np.int32(imgpts1).reshape(-1,2)
            imgpts2, jac= cv2.projectPoints(axis, rvec[1], tvec[1], mtx, dist)
            imgpts2 = np.int32(imgpts2).reshape(-1,2)
            cv2.line(frame, tuple(imgpts1[0].ravel()), tuple(imgpts2[0].ravel()), color, int(thickness/2))
            center1 = imgpts1[0].ravel()
            center2 = imgpts2[0].ravel()
            text = f"d = {distance:.2f}m"
 
            text_position_x = center1[0]+20 + (center2[0]-center1[0])//2
            text_position_y = center1[1] + (center2[1]-center1[1])//2
            text_position = tuple([text_position_x, text_position_y])
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color,2,cv2.LINE_AA)

        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+' '

        cv2.putText(frame, "Id(s): " + strg, (20,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

    else:
        cv2.putText(frame, "No Ids", (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release
cap.release()
cv2.destroyAllWindows()