import numpy as np
import cv2 as cv
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_7X7_50)
aruco_params = cv.aruco.DetectorParameters_create()

tag_corners = np.zeros((4,2))
CORNER_POS_LIST = [[24,24],[624,24],[624,460],[24,460]] # The positions of the AR tag outer corners in transformed space
FRAME_SIZE = (648,484)
FRAMES_GONE_THRESHOLD = 5

# Track whether the AR tags have been gone for several frames, so brief loss of tags doesn't cause issues
frames_gone = FRAMES_GONE_THRESHOLD
work_frames_recorded = 0
pattern_found = False
pattern_frame = None
stitch_pattern = np.zeros((81, 122))

# Updates the corner entries in the list of points for the homography matrix
def update_corners(corners, ids):
    for (corner, id) in zip(corners, ids):
        corners_list = corner.reshape((4,2))

        # extract the correct outer corner based on ARuco index
        tag_corners[id-44] = corners_list[id-44]

# Transforms and crops the frame to the cross-stitch work area
def transform_work_area(frame, padding):
    pts_dst = np.array(CORNER_POS_LIST)
    h, status = cv.findHomography(tag_corners, pts_dst)
    frame = cv.warpPerspective(frame, h, FRAME_SIZE) # img size is based on 4 px per stitch
    frame = frame[padding:-padding, padding:-padding] # crop to just get working area
    return h, frame

def get_patterned_frame(frame):
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

        # print(np.where(frame==0))
        
        frame_orig_size = cv.resize(pattern_frame, (0,0), fx=4, fy=4, interpolation=cv.INTER_NEAREST)
        frame_bordered = cv.copyMakeBorder(frame_orig_size, 80, 80, 80, 80, cv.BORDER_CONSTANT, value=255)
        return frame_bordered
    return None

while True:
    # Capture frame-by-frame
    ret, orig_frame = cap.read()
    frame = orig_frame
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    (corners, ids, rejected) = cv.aruco.detectMarkers(frame, aruco_dict,
	parameters=aruco_params)
    if ids is not None and ids.shape[0] > 0:
        ids = ids.flatten()
        update_corners(corners, ids)
        if np.count_nonzero(tag_corners) < 8: # don't start tracking until we've seen all four corners
            continue
        
        # Transform and threshold work area
        h_mat, work_frame = transform_work_area(frame, 80)
        gray = cv.cvtColor(work_frame, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)

        # dilate to make it easier to detect the correct spots for the pattern
        kernel = np.ones((1, 1), np.uint8)
        thresh_dilated = cv.dilate(thresh, kernel)
        # print(thresh_dilated.shape)
        work_resized = cv.resize(thresh_dilated, (122, 81), interpolation=cv.INTER_NEAREST) # downsample to 1 px/stitch
        if work_frames_recorded < 30:
            stitch_pattern = np.add(stitch_pattern, work_resized)
            work_frames_recorded += 1
        elif pattern_found == False:
            stitch_pattern = np.round(stitch_pattern/(30*255))*255
            cv.imshow("Stitch pattern", cv.resize(stitch_pattern, (0,0), fx=8, fy=8, interpolation=cv.INTER_NEAREST))
            pattern_found = True
            pattern_frame = get_patterned_frame(work_resized)

        # print(work_resized.shape)

        if pattern_frame is not None:
            h_inverse = np.linalg.inv(h_mat)
            pattern_frame_orig = cv.warpPerspective(pattern_frame, h_inverse, (1280, 720), borderMode = cv.BORDER_CONSTANT, borderValue=255)
            orig_frame = cv.bitwise_and(orig_frame, orig_frame, mask=pattern_frame_orig)
        
        frames_gone = 0
    else:
        frames_gone += 1
        if frames_gone >= FRAMES_GONE_THRESHOLD: # If we've lost the work piece for a long enough time
            pattern_found = False
            work_frames_recorded = 0

    cv.imshow("Detected Circle", orig_frame)
    # cv.waitKey(0)
    # cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
