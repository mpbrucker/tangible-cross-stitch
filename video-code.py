import numpy as np
import cv2 as cv
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_7X7_50)
aruco_params = cv.aruco.DetectorParameters_create()

tag_corners = np.zeros((4,2))

def update_homography(corners, ids):
    for (corner, id) in zip(corners, ids):
        corners_list = corner.reshape((4,2))
        (top_left, top_right, bottom_right, bottom_left) = corners_list

        # extract the correct outer corner based on ARuco index
        tag_corners[id-44] = corners_list[id-44]

    


while True:
    # Capture frame-by-frame
    ret, orig_frame = cap.read()
    frame = orig_frame
    print(orig_frame.shape)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (corners, ids, rejected) = cv.aruco.detectMarkers(gray, aruco_dict,
	parameters=aruco_params)
    if ids is not None and ids.shape[0] > 0: # TODO change to ids.shape[0] > 0 
        ids = ids.flatten()
        update_homography(corners, ids)
        if np.count_nonzero(tag_corners) < 8: # don't start tracking until we've seen all four corners
            continue
        
        # pts_dst = np.array([[0,0], [200,0], [200,200], [0, 200]])
        # pts_dst = np.array([[24,24],[60,24],[60,60],[24,60]])
        pts_dst = np.array([[24,24],[624,24],[624,460],[24,460]])
        h, status = cv.findHomography(tag_corners, pts_dst)
        frame = cv.warpPerspective(frame, h, (648,484)) # img size is based on 4 px per stitch
        frame = frame[80:404, 80:568] # crop to just get working area

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, frame = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)

        # dilate to make it easier to detect the correct spots for the pattern
        kernel = np.ones((1, 1), np.uint8)
        frame = cv.dilate(frame, kernel)


        frame = cv.resize(frame, (122, 81), interpolation=cv.INTER_NEAREST) # downsample to 1 px/stitch


        pattern_x, pattern_y = np.where(frame==0)
        if len(pattern_x) > 0:
            pattern_area = frame[min(pattern_x)-1:max(pattern_x)+1, min(pattern_y)-1:max(pattern_y)+1]
            y_rep = int(frame.shape[0]/pattern_area.shape[0])+2
            x_rep = int(frame.shape[1]/pattern_area.shape[1])+2



            pattern_frame = np.tile(pattern_area, (y_rep,x_rep))
            y_offset = pattern_area.shape[0]-(min(pattern_y)%pattern_area.shape[0])
            x_offset = pattern_area.shape[1]-(min(pattern_x)%pattern_area.shape[1])
            # print(x_offset, y_offset)
            pattern_frame = pattern_frame[y_offset:y_offset+81,0:0+122]

            frame = pattern_frame      

            # print(np.where(frame==0))
            
            frame = cv.resize(frame, (0,0), fx=4, fy=4, interpolation=cv.INTER_NEAREST)
            frame = cv.copyMakeBorder(frame, 80, 80, 400-frame.shape[0], 560-frame.shape[1], cv.BORDER_CONSTANT, value=255)
            h_inverse = np.linalg.inv(h)
            frame = cv.warpPerspective(frame, h_inverse, (1280, 720))
            orig_frame = cv.bitwise_and(orig_frame, orig_frame, mask=frame)


    cv.imshow("Detected Circle", orig_frame)
    # cv.waitKey(0)
    # cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
