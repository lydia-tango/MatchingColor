import numpy
import cv2
import cvk2

# createMask: create a mask by threshold an image
# @param: image, color to be thresholded, threshVal, maxVal, threshType
# @return: mask of the image, bool mask, and thresh
def createMask(image, color, threshVal, maxVal, threshType):
		myimage = image.copy()
		h = myimage.shape[0]
		w = myimage.shape[1]
		image_float = myimage.astype(float)
		dists_float = image_float - numpy.tile(color, (h, w, 1))
		dists_float = dists_float * dists_float
		dists_float = dists_float.sum(axis=2)
		dists_float = numpy.sqrt(dists_float)
		dists_uint8 = numpy.empty(dists_float.shape, 'uint8')
		cv2.convertScaleAbs(dists_float, dists_uint8, 1, 0)
		mask = numpy.zeros((h, w), 'uint8')
		bmask = mask.view(numpy.bool)
		ret, thresh = cv2.threshold(dists_uint8, threshVal, maxVal, threshType, mask)

		return mask, bmask, thresh





imageFilenameRoot = "img/lg-"
win = 'Project 2'
cv2.namedWindow(win)

for i in range(1000,1060):
	image_rgb = cv2.imread(imageFilenameRoot + str(i)+".jpg")
	h = image_rgb.shape[0]
	w = image_rgb.shape[1]



	img_large = numpy.empty((h + 30, w + 30, 3), 'uint8')
	img_large[:] = (255, 255, 255)
	img_large[15:15+h,15:15+w]=image_rgb

	h = h + 30
	w = w + 30
	image_rgb = img_large.copy()
	display_gray = numpy.empty((h, w), 'uint8')
	cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY, display_gray)

	th2 = cv2.adaptiveThreshold(display_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
	cv2.imshow('Project 2', th2)
	while cv2.waitKey(15) < 0: pass
	cv2.imshow('Project 2', img_large)
	while cv2.waitKey(15) < 0: pass
	# cv2.imwrite('pattern' + str(i) + '.png', th2[0:256,0:256])


	# kernel = numpy.ones((1,1), numpy.uint8)
	# th2 = cv2.dilate(th2, kernel, iterations = 1)
	# th2 = cv2.erode(th2, kernel, iterations = 1)

	# cv2.imshow('Project 2', th2)
	# while cv2.waitKey(15) < 0: pass

	contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	print len(contours)
	largeContour = []
	largestContour = []
	for i in range(len(contours)):
		if len(contours[i]) > 200: 
			largeContour.append(contours[i])
			if len(contours[i]) > len(largestContour):
				largestContour = contours[i]


	print len(largeContour), len(largestContour)
	for c in largeContour:
		area = cv2.contourArea(c)
	# M = cv2.moments(argestContour)
	# cx = int(M['m10']/M['m00'])
	# cy = int(M['m01']/M['m00'])
		info = cvk2.getcontourinfo(c)
		mu = info['mean']
		mask = numpy.zeros(image_rgb.shape,numpy.uint8)
		cv2.drawContours(mask,[c],0,(255, 255,255),-1)

		
		# cv2.circle(image_rgb,(int(mu[0]),int(mu[1])), 4, (0,0,255), 1)
		bmask = mask.view(numpy.bool)
		# mask1, bmask1, thresh1 = createMask(mask, (255, 0, 0), 30, 255, cv2.THRESH_BINARY_INV)

		# cv2.imshow('Project 2', mask)
		# while cv2.waitKey(15) < 0: pass

		img = image_rgb.copy()
		img[bmask] = image_rgb[bmask]
		# cv2.imshow('Project 2', img )
		# while cv2.waitKey(15) < 0: pass


# get only the picture of the clothes


# do adaptive threshold

# ge multiple contours

# if too much contours --> patterns

# if not --> single color

# get mean color in each contour




# get 10% of the pixels of pictures in each category

# do kmeans

# evaluate the colors and make a histogram for each picture

# do p-means to find the mean color scheme






