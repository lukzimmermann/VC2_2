import cv2
import numpy as np
content = cv2.imread("IMG_0501.png")  # Load your content here
cap = cv2.VideoCapture(1)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

camera_matrix = np.array([[5504.1, 0, 2001.1],
                          [0, 5591.4, 1900.8],
                          [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.array([0.04644, -0.04812, 0.0952, 0.000425, 1.144], dtype=np.float64)


while True:
    ret, frame = cap.read()
    
    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)
    
    if ids is not None:
        for i in range(ids.size):
            # Estimate marker pose
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
            
            # Project virtual content onto the marker
            # ...
            
        # Draw markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    cv2.imshow("AR using ArUco", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()