import numpy as np
import cv2
import glob

######################################################################################################################
# Section 1: Following section defines the data structure and parameters used in the file globally
######################################################################################################################
# Following coordinates are used to define the perspective transform's Source and Destination points
src = np.float32(
    [
        [1210, 710],  # Bottom right
        [160, 710],  # Bottom left
        [580, 460],  # Top left
        [735, 460]  # Top right
    ])

dst = np.float32(
    [
        [1020, 720],  # Bottom right
        [280, 720],  # Bottom left
        [280, 1],  # Top left
        [1020, 1],  # Top right
    ])

# Following class is used to store the data lane detection for frame
class Line:
    def __init__(self):
        self.detected = False

        self.left_fitx = np.array([])
        self.right_fitx = np.array([])

        self.ploty = np.array([])

        self.left_fit = np.array([])
        self.right_fit = np.array([])

        self.leftXHist_base = 0
        self.rightXHist_base = 0

        self.radius_of_curvature = 0
        self.veh_position_from_center = 0

        self.badFrameCount = 0

    # Following class methods are defined to optimize the use of data elements
    def getPreviousData(self):
        return self.left_fitx, self.right_fitx, self.ploty, self.left_fit, self.right_fit

    def wasLaneDetectedInLastFrame(self):
        return self.detected

    def getRadiusOfCurvature(self):
        return self.radius_of_curvature

    def getVehiclePositionFromCenter(self):
        return self.veh_position_from_center

    def getLastHistData(self):
        return self.leftXHist_base, self.rightXHist_base

    def getBadFrameCount(self):
        return self.badFrameCount

    def updateBadFrameCount(self, badFrameCount):
        self.badFrameCount = badFrameCount

    def saveRadiusOfCurvature(self, radius):
        self.radius_of_curvature = radius

    def saveVehiclePositionFromCenter(self, position):
        self.veh_position_from_center = position

    def saveHistBase(self, leftx_base, rightx_base):
        self.leftXHist_base = leftx_base
        self.rightXHist_base = rightx_base

    def updateDetectedSave(self, detected):
        self.detected = detected

    def savePolyData(self, left_fitx, right_fitx, ploty, left_fit, right_fit):
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.ploty = ploty
        self.left_fit = left_fit
        self.right_fit = right_fit

    def resetLaneData(self):
        self.detected = False
        self.left_fitx = np.array([])
        self.right_fitx = np.array([])
        self.ploty = np.array([])
        self.left_fit = np.array([])
        self.right_fit = np.array([])
        self.leftXHist_base = 0
        self.rightXHist_base = 0
        self.radius_of_curvature = 0
        self.badFrameCount = 0

# Creating the instance of the class
lastFrameData = Line()

######################################################################################################################
# Section 2: Following section defines all the utility functions required to complete the project
######################################################################################################################

# Calibrate Camera method uses different chess board images to calculate the camera matrix which will
# be used to undistort the given image.
def calibrateCamera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = img.shape[0:2]

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return ret, mtx, dist, rvecs, tvecs

ret, mtx, dist, rvecs, tvecs = calibrateCamera()
# --------------------------------------------------------------------------------------------------------------------#

# This method takes the camera matrix from CalibrateCamera method and returns the undistorted image
def undistortImage(img):
    return cv2.undistort(img, mtx, dist, None, mtx)
# --------------------------------------------------------------------------------------------------------------------#
# This method converts the image to LAB color space and separate the B channel
# It also performs the thresholding and returns the binary image
def getBChannelImage(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    b_binary = np.zeros_like(b)
    b_binary[(b > thresh[0]) & (b <= thresh[1])] = 1
    return b_binary

# --------------------------------------------------------------------------------------------------------------------#
# This method converts the image to LUV color space and separate the L channel
# It also performs the thresholding and returns the binary image
def getLChannelImage(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
    l, u, v = cv2.split(luv)
    l_binary = np.zeros_like(l)
    l_binary[(l > thresh[0]) & (l <= thresh[1])] = 1
    return l_binary
# --------------------------------------------------------------------------------------------------------------------#

# This method takes 2 binary images and combines them to create a combined image
# This is an important part of project, because only single transformed image is not sufficient
def combineBinaryImages(img1, img2):
    combined = np.zeros_like(img1)
    combined[(img1 == 1) | (img2 == 1)] = 1

    return combined
# --------------------------------------------------------------------------------------------------------------------#

# This method extracts the given area in an image and performs the perspective transform to the given destination points
# It also returns the perspective and inverse perspective matrix, which are important parameters to
# redraw the selected area on the final image. It also returns the warped image
def perspectiveTransform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    # Warp the image using OpenCV warpPerspective()
    ysize, xsize = img.shape[0:2]
    warped = cv2.warpPerspective(img, M, (xsize,ysize), flags=cv2.INTER_LINEAR)
    # Return the resulting image and matrix
    return warped, M, Minv
# --------------------------------------------------------------------------------------------------------------------#

# This method returns the histogram of the given image, since we are interested in only lower half of the image so
# the shape of the image is divided by 2
def getHistogram(warpedCombine):
    return np.sum(warpedCombine[warpedCombine.shape[0]//2:,:], axis=0)
# --------------------------------------------------------------------------------------------------------------------#

# This method returns a blank image of the given image's shape
def createImageAs(warpedCombine):
    return np.dstack((warpedCombine, warpedCombine, warpedCombine))*255
# --------------------------------------------------------------------------------------------------------------------#

#  This method performs the sliding window search on the given image and tries to detect the lane lines
#  It finds the different non zero coordinates in X and Y direction in the given window +- margin
def slidingWindowSearch(warpedCombine, out_img, leftx_base, rightx_base, numOfWindows=9, widthOfWin=200,
                        detectParam=50):
    # Defining the number of sliding windows
    nwindows = numOfWindows
    # Setting the height of windows
    window_height = np.int(warpedCombine.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warpedCombine.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Setting the width of the windows +/- margin
    margin = np.int(widthOfWin / 2)

    # Setting the minimum number of pixels found to recenter window
    minpix = detectParam

    # Creating empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Stepping through the windows one by one
    for window in range(nwindows):
        # Defining window boundaries in x and y (and right and left)
        win_y_low = warpedCombine.shape[0] - (window + 1) * window_height
        win_y_high = warpedCombine.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identifying the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]

        # Appending these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If we found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return left_lane_inds, right_lane_inds, nonzerox, nonzeroy
# --------------------------------------------------------------------------------------------------------------------#

# This method performs the search for lane in the previous frame's identified region
# Once we know the lane position in one frame then we can perform the concentrated search
# on the previous detected area
def knownLaneDetection(warpedCombine, left_fit, right_fit):

    nonzero = warpedCombine.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    return left_lane_inds, right_lane_inds, nonzerox, nonzeroy
# --------------------------------------------------------------------------------------------------------------------#

# This method tries to fit the given order polynomial to find the X and Y regions on image indies
# Image indies are generated as output of slidingWindowSearch or knownLaneDetection method
def fitPolynomial(warpedCombine, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, polyOrder=2):
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a given order polynomial to each
    left_fit = np.polyfit(lefty, leftx, polyOrder)
    right_fit = np.polyfit(righty, rightx, polyOrder)

    # Generating x and y values for plotting -  Ay^2 + By + C
    ploty = np.linspace(0, warpedCombine.shape[0] - 1, warpedCombine.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx
# --------------------------------------------------------------------------------------------------------------------#

# This method returns the curvature of the lanes based on the given Left and Right coordinates
def getCurvature(leftx, rightx, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 25 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 720  # meters per pixel in x dimension

    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    #(ref: http://www.intmath.com/applications-differentiation/8-radius-curvature.php)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad
# --------------------------------------------------------------------------------------------------------------------#

# This method finds the vehicle position from the lane center and saves the drifted postion in
# Class Line's instance 'lastFrameData'
def handleVehPositionFromCenter(leftx_base, rightx_base, imageMidpoint):

    xm_per_pix = 3.7 / 740  # meters per pixel in x dimension
    detectedCenter = (leftx_base + rightx_base) / 2
    position = (imageMidpoint - detectedCenter)
    positionInMtr = position * xm_per_pix
    # Saving the data to class instance
    lastFrameData.saveVehiclePositionFromCenter(positionInMtr)
# --------------------------------------------------------------------------------------------------------------------#

# This method implements the algorithm to detect lane in the given image
def findLaneLines(histogram, warpedCombine, out_img, numOfWindows=9, widthOfWin=200, detectParam=50):
    # Calculate the image mid point
    midpoint = np.int(histogram.shape[0] / 2)
    # Find the left and right max intensity postions in X direction
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Find vehicle postion from center and save in Class instance "lastFrameData"
    handleVehPositionFromCenter(leftx_base, rightx_base, midpoint)

    newLeftBase = leftx_base
    newRightBase = rightx_base

    # IF there there was no lane detected or if there were bad frames for continuous 8 times then
    # perform Sliding window search to search lane from scratch
    if (lastFrameData.wasLaneDetectedInLastFrame()):
        # Retreive stored data from class instance
        p_left_fitx, p_right_fitx, p_ploty, p_left_fit, p_right_fit = lastFrameData.getPreviousData()
        p_leftx_base, p_rightx_base = lastFrameData.getLastHistData()

        # These variables determine if right or left lane was not detected in the given margin of the
        # old position of the detected lane from previous frame
        lSimilarToPrev, rSimilarToPrev = False, False

        # Search and update the flags if the new lanes are detected in the given margin from the previous frame
        # If the left or right lane was not the same from previous frame then retain the previous frame's
        # data and increment the bad frame count
        if (np.abs(p_leftx_base - leftx_base) <= 80):
            lSimilarToPrev = True
        else:
            newLeftBase = p_leftx_base

        if (np.abs(p_rightx_base - rightx_base) <= 80):
            rSimilarToPrev = True
        else:
            newRightBase = p_rightx_base

        # If both lanes were not detected within margin area of the previous frame, then perform Sliding window
        # to search lanes from the scratch, also fit the polinomial of the given order
        if ((lSimilarToPrev == False) and (rSimilarToPrev == False)):
            lastFrameData.updateDetectedSave(True)
            lastFrameData.updateBadFrameCount(0)

            left_lane_inds, right_lane_inds, nonzerox, nonzeroy = slidingWindowSearch(warpedCombine, out_img,
                                                                                      newLeftBase, newRightBase,
                                                                                      numOfWindows, widthOfWin,
                                                                                      detectParam)
            newleft_fitx, newright_fitx, newploty, newleft_fit, newright_fit, newleftx, newrightx = fitPolynomial(
                warpedCombine,
                left_lane_inds,
                right_lane_inds, nonzerox,
                nonzeroy, 2)
        # If any one or both the lanes were detected at same postion +/- margin of the previous frame,
        # Then perform the search in the previous frame's reagion and
        # also update the bad frame flag accordingly, also fit the polinomial of the given order
        else:
            newleft_fit = p_left_fit
            newright_fit = p_right_fit

            left_lane_inds, right_lane_inds, nonzerox, nonzeroy = knownLaneDetection(warpedCombine, newleft_fit,
                                                                                     newright_fit)
            newleft_fitx, newright_fitx, newploty, newleft_fit, newright_fit, newleftx, newrightx = fitPolynomial(
                warpedCombine,
                left_lane_inds,
                right_lane_inds, nonzerox,
                nonzeroy, 2)

            if (lSimilarToPrev == False):
                newleft_fitx = p_left_fitx
                newleft_fit = p_left_fit
                lastFrameData.updateBadFrameCount(lastFrameData.getBadFrameCount() + 1)

            if (rSimilarToPrev == False):
                newright_fitx = p_right_fitx
                newright_fit = p_right_fit
                lastFrameData.updateBadFrameCount(lastFrameData.getBadFrameCount() + 1)

            if ((lSimilarToPrev == True) and (rSimilarToPrev == True)):
                lastFrameData.updateBadFrameCount(0)

            if (lastFrameData.getBadFrameCount() > 8):
                lastFrameData.updateDetectedSave(False)

    else:
        lastFrameData.updateDetectedSave(True)
        lastFrameData.updateBadFrameCount(0)

        left_lane_inds, right_lane_inds, nonzerox, nonzeroy = slidingWindowSearch(warpedCombine, out_img, newLeftBase,
                                                                                  newRightBase, numOfWindows,
                                                                                  widthOfWin,
                                                                                  detectParam)
        newleft_fitx, newright_fitx, newploty, newleft_fit, newright_fit, newleftx, newrightx = fitPolynomial(
            warpedCombine, left_lane_inds,
            right_lane_inds, nonzerox,
            nonzeroy, 2)

    # Find and update the Class instance with the Lane curvature
    leftC, RightC = getCurvature(newleft_fitx, newright_fitx, newploty)
    lastFrameData.saveRadiusOfCurvature((leftC + RightC) / 2)
    # Save the data to Class instance for further processing on next frames
    lastFrameData.savePolyData(newleft_fitx, newright_fitx, newploty, newleft_fit, newright_fit)
    lastFrameData.saveHistBase(newLeftBase, newRightBase)

    return newleft_fitx, newright_fitx, newploty, newleft_fit, newright_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy
# --------------------------------------------------------------------------------------------------------------------#

# This method performs the inverse perspective transform and fills the color on the
# final image on the given detected areas
def fillFinalImage(undistortRgb, warpedCombine, left_fitx, right_fitx, ploty, Minv):
    # Creating an image to draw the lines on
    warp_zero = np.zeros_like(warpedCombine).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recasting the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Drawing the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warping the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistortRgb.shape[1], undistortRgb.shape[0]))
    # Combining the result with the original image
    result = cv2.addWeighted(undistortRgb, 1, newwarp, 0.3, 0)
    return result
# --------------------------------------------------------------------------------------------------------------------#

# This method writes the given text on the given image at the given coordinate
def writeOnImage(img, text, coOrd = (100,150)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(img, text, coOrd, font, 1,(255,255,255),2,cv2.LINE_AA)
# --------------------------------------------------------------------------------------------------------------------#

# This method performs the processing of the image to write different data on the final image.
def processForDetails(outputImg, undistortimage):
    # Retrieve the radius of curvature from the class instance, and write on image
    radius = lastFrameData.getRadiusOfCurvature()
    radius = float("{0:.3f}".format(radius))
    writeOnImage(outputImg, "Radius Of Curvature: " + str(radius), coOrd=(100, 200))

    # Retrieve the vehicle position from the center and write on the image
    vehPosition = lastFrameData.getVehiclePositionFromCenter()
    vehPosition = float("{0:.3f}".format(vehPosition))

    if (vehPosition > 0):
        writeOnImage(outputImg, "Vehicle Position From Center: " + str(vehPosition) + 'm (Right Side)',
                     coOrd=(100, 240))
    elif (vehPosition < 0):
        writeOnImage(outputImg, "Vehicle Position From Center: " + str(vehPosition) + 'm (Left Side)', coOrd=(100, 240))
    else:
        writeOnImage(outputImg, "Vehicle Position From Center: " + str(vehPosition) + 'm (+Center+)', coOrd=(100, 240))

    # Generate the undistorted warped image to embbed on the final image to show
    # real processed image for better understanding
    warped, Mlocal, Minvlocal = perspectiveTransform(undistortimage, src, dst)
    resized = cv2.resize(warped, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    outputImg[0:resized.shape[0], 0:resized.shape[1]] = resized

    writeOnImage(outputImg, "<- Perspective Transformed Area", coOrd=(300, 100))
    return outputImg

#################################################################################################################
# This method consists of the real pipeline created for finding lane lines, it takes the image and performs
# series of operations and returns the final, filled and processed image
#################################################################################################################
def findLaneLinesPipeline(inputImg):
    # Undistort image
    undistortimage = undistortImage(inputImg)
    # Extracting the B channel binary
    bBinary = getBChannelImage(cv2.cvtColor(undistortimage, cv2.COLOR_BGR2RGB), (145, 255))
    #Extracting the L channel binary
    LBinary = getLChannelImage(undistortimage, (210, 255))
    #Combining 2 binary images
    combine = combineBinaryImages(LBinary, bBinary)
    # Performing perspective transform
    warpedCombine, M, Minv = perspectiveTransform(combine, src, dst)
    # Getting the histogram of the combined
    histogram = getHistogram(warpedCombine)
    # Creating an output image to draw on and  visualize the result
    out_img = createImageAs(warpedCombine)
    # Executing lane line finding algorithm
    left_fitx, right_fitx, ploty, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = findLaneLines(
        histogram, warpedCombine, out_img)
    # Filling the final image's given region
    outputImg = fillFinalImage(undistortimage, warpedCombine, left_fitx, right_fitx, ploty, Minv)
    # Processing image to write details on it
    outputImg = processForDetails(outputImg, undistortimage)
    return outputImg

# Following code processes the given video's each frame with the LaneLine finding pipeline and creates the
# resulted video
# from moviepy.editor import VideoFileClip
#
# def createOutVideo():
#     white_output = 'output.mp4'
#     clip1 = VideoFileClip("project_video.mp4")
#     white_clip = clip1.fl_image(findLaneLinesPipeline)
#     white_clip.write_videofile(white_output, audio=False)
#
# lastFrameData.resetLaneData()
# createOutVideo()
