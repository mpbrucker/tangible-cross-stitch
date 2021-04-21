import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_7X7_50)
aruco_params = cv.aruco.DetectorParameters_create()
while True:
    # Capture frame-by-frame
    ret, orig_frame = cap.read()
    frame = orig_frame
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (corners, ids, rejected) = cv.aruco.detectMarkers(gray, aruco_dict,
	parameters=aruco_params)
    tag_corners = np.zeros((4,2))
    if ids is not None and ids.shape[0] == 4:
        ids = ids.flatten()
        for (corner, id) in zip(corners, ids):
            corners_list = corner.reshape((4,2))
            (top_left, top_right, bottom_right, bottom_left) = corners_list

            # extract the correct outer corner based on ARuco index
            tag_corners[id-44] = corners_list[id-44]

            # cv.line(frame, tuple(top_left), tuple(top_right), (0, 255, 0), 2)
            # cv.line(frame, tuple(top_right), tuple(bottom_right), (0, 255, 0), 2)
            # cv.line(frame, tuple(bottom_right), tuple(bottom_left), (0, 255, 0), 2)
            # cv.line(frame, tuple(bottom_left), tuple(top_left), (0, 255, 0), 2)
        # pts_dst = np.array([[24,24],[60,24],[60,60],[24,60]])
        pts_dst = np.array([[24,24],[624,24],[624,460],[24,460]])
        # pts_dst = np.array([[0,0], [200,0], [200,200], [0, 200]])
        h, status = cv.findHomography(np.array(tag_corners), pts_dst)
        frame = cv.warpPerspective(frame, h, (648,484)) # img size is based on 4 px per stitch
        frame = frame[80:404, 80:568] # crop to just get working area
        frame = cv.resize(frame, (122, 81), interpolation = cv.INTER_NEAREST) # downsample to 1 px/stitch

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, frame = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
        pattern_x, pattern_y = np.where(frame==0)
        if len(pattern_x) > 0:
            # print([min(pattern_x), max(pattern_x), min(pattern_y), max(pattern_y)]) 
            pattern_area = frame[min(pattern_x)-1:max(pattern_x)+1, min(pattern_y)-1:max(pattern_y)+1]
            x_rep = int(frame.shape[0]/pattern_area.shape[0])
            y_rep = int(frame.shape[1]/pattern_area.shape[1])
            pattern_frame = np.tile(pattern_area, (x_rep,y_rep))
            frame = pattern_frame      

            # print(np.where(frame==0))
            
            frame = cv.resize(frame, (0,0), fx=4, fy=4, interpolation=cv.INTER_NEAREST)
            frame = cv.copyMakeBorder(frame, 80, 80, 400-frame.shape[0], 560-frame.shape[1], cv.BORDER_CONSTANT, value=255)
            h_inverse = np.linalg.inv(h)
            frame = cv.warpPerspective(frame, h_inverse, (640, 480))
            orig_frame = cv.bitwise_and(orig_frame, orig_frame, mask=frame)


    # gray_blurred = cv.blur(gray, (5, 5))
    # Display the resulting frame

    # detected_circles = cv.HoughCircles(gray_blurred, 
            #        cv.HOUGH_GRADIENT, 1, 50, param1 = 100,
            #    param2 = 90, minRadius = 75, maxRadius = 200)
  
    # Draw circles that are detected.
    # if detected_circles is not None:
    
    #     # Convert the circle parameters a, b and r to integers.
    #     detected_circles = np.uint16(np.around(detected_circles))
    
    #     for pt in detected_circles[0, :]:
    #         a, b, r = pt[0], pt[1], pt[2]

    #         cv.putText(frame, "Hello World!", (a-75, b), cv.FONT_HERSHEY_SIMPLEX, 1, (0,250,0,255), 3)
            # Draw the circumference of the circle.
            # cv.circle(frame, (a, b), r, (0, 255, 0), 2)
    
            # # Draw a small circle (of radius 1) to show the center.
            # cv.circle(frame, (a, b), 1, (0, 0, 255), 3)
    cv.imshow("Detected Circle", orig_frame)
    # cv.waitKey(0)
    # cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
