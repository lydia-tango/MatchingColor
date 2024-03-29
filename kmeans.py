import numpy as np
import cv2

allpixel = None


for i in range(1000, 1002):
	img = cv2.imread("img/lg-" + str(i) +".jpg")
	cv2.imshow('res2',img)
	cv2.waitKey(0)
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)
	if allpixel == None:
		allpixel = Z
	else:
		allpixel = numpy.concatenate((allpixel, Z))

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 10
	ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
 	res = center[label.flatten()]
	res2 = res.reshape((img.shape))

	cv2.imshow('res2',res2)
	cv2.waitKey(0)
cv2.destroyAllWindows()