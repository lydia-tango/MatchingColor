import cv2
import numpy
# where we define the category of men clothes
from defClothes import *
from matplotlib import pyplot as plt


def getPixels(targetFile, imageFilenameRoot, noiseFile):
	# targetFile stores all the category information of the images
	target = open(targetFile, "r")
	# noiseFile stores all the index of noise data 
	noise = open(noiseFile, "r")


	# diversity = len(article_text)
	# num_list = [0] * diversity
	noise_list = []
	article_index = 4
	K = 5


	# add all indeces of noices into noise_list
	while 1:
		line = noise.readline().split()
		if len(line) == 0: break
		noise_list.append(line[0])


	allpixel = numpy.empty((1,1,1), 'uint8')	# all pixel is a n * 3 matrix of all n pixels in images
	num_suits = 0

	# loop through images 
	while 1:
		
		line = target.readline().split()
		if len(line) == 0: break
		
		# ignore noise images
		if line[0] in noise_list:
			# print "noise"
			continue


		if line[1].lower() in article_list[article_index]:
			num_suits +=1

			image_rgb = cv2.imread(imageFilenameRoot + line[0]+".jpg")

			h = 8
			w = 6
			image_rgb = cv2.resize(image_rgb, (w,h), image_rgb, 0, 0, cv2.INTER_LANCZOS4)	

			# construct a wh * 3 matrix of all pixels in the image
			image_pixels = image_rgb.reshape((-1,3))
			# convert to np.float32
			image_pixels = numpy.float32(image_pixels)
			# append the image_pixels to a large matrix
			if allpixel.size < 10:
				allpixel = image_pixels
			else:
				allpixel = numpy.concatenate((allpixel, image_pixels))

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# print allpixel.shape
	ret,label,center=cv2.kmeans(allpixel,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)	
	center_color_display = numpy.empty((30, K * 30, 3), 'uint8')
	for i, c in enumerate(center):
		center_color_display[:, i*30: (i+1)*30] = c
	cv2.imwrite('analysis/' + article_text[article_index] + str(K) + '.jpg', center_color_display)
	# cv2.imshow('res2', center_color_display)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	print "Finished kmeans round 1 with", num_suits, article_text[article_index]

	label_each_image_list = numpy.vsplit(label, num_suits)
	allHist = numpy.empty((1,1,1), 'float32')
	for label_each_image in label_each_image_list:
		# label_matrix = label_each_image[:, :, numpy.newaxis]
		# label_matrix = label_matrix.astype('uint8') 

		# Now convert back into uint8, and make original image
		center = numpy.uint8(center)
		res = center[label_each_image.flatten()]
		res2 = res.reshape(h, w, 3)
		# get a histogram of color distribution of each image
		hist, bins = numpy.histogram(label_each_image, K, [0, K])
		hist = hist.astype("f")
		hist /= h*w # normalize distribution
		if allHist.size < 4:
			allHist = hist
		else:
			allHist = numpy.vstack((allHist, hist))
	# print allHist

	# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# print allpixel.shape
	P = 10
	histRet,histLabel,histCenter=cv2.kmeans(allHist,P,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	print histCenter
	# print histRet
	# print histLabel	
	# print hist.type

		# plt.plot(hist)
		# plt.xlim([0,K])
		# plt.show()

		
		# ------------------------------ start working again from here --------------------------------
		#create a histogram for every image
		# channels = [0, 1, 2]
		# allChHist = None
		# channels = cv2.split(res2)

		# loop over the image channels
		# color = ('b','g','r')
		# for i,col in enumerate(color):
		# print res2.shape
		# print label_matrix
		# histr = cv2.calcHist(label_matrix,[0],None,[K],[0, 3])

		# for channel in channels: # create a histogram each channel

		# 	hist = cv2.calcHist([channel], [0], None, [K], [0, K]) # (256L, 1L)
		# 	hist /= h*w
		# 	print "ind. hist size:", hist.shape

		# 	#want to normalize now -> really not sure if this is working
		# 	cv2.normalize(hist, hist, 1)

		# 	# concatenate the histograms for each color channel
		# 	if allChHist == None:
		# 		allChHist = hist
		# 	else:
		# 		allChHist = numpy.concatenate((allChHist, hist)) #ends up as (768L, 1L)
		# print "allChHist shape: ", allChHist.shape
		 
 

		

	#run p-means to get "representative" color schemes
	#p~10





	# 	for index in range(diversity):
	# 		# if index in [3, 4, 5, 6, 7, 8]:
	# 		# 	max_num = 210
	# 		# else:
	# 		# 	max_num = 1000


	# 		if line[1].lower() in article_list[index]:
	# 			if num_list[index] <= max_num:
	# 				output = "0 " * index + "1 " + "0 " * (diversity-index-1)
	# 				outfile.write( output + "\n")
	# 				num_list[index] = num_list[index] + 1

	# 				image_rgb = cv2.imread(imageFilenameRoot + line[0]+".jpg")
	# 				h = 100
	# 				w = 75

	# 				image_rgb = cv2.resize(image_rgb, (w,h), image_rgb, 0, 0, cv2.INTER_LANCZOS4)
	# 				edges = cv2.Canny(image_rgb, 100, 200)

	# 				# smallImg = smallImg.astype(float)
	# 				# smallGray = numpy.empty((h,w), 'uint8')
	# 				image = image_rgb
	# 				# cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, smallGray)
	# 				#smallImg = numpy.divide(smallGray*1.0, 255.0)

	# 				for x in range(image.shape[0]):
	# 					for y in range(image.shape[1]):
	# 						pixel = float(edges[x][y])/255.0
	# 						infile.write("%.4f " % pixel)

	# 						for c in range(image.shape[2]):
	# 							color = float(image[x][y][c])/255.0
	# 							# if i:
	# 							# 	print type(image[x][y][c])
	# 							# 	print int(image[x][y][c])

	# 							# 	i -= 1

	# 							infile.write("%.4f " % color)
							
	# 				infile.write("\n")

	# 				if verbose:
	# 					print article_text[index], output, line
	# 				break


		
	# for i in range(diversity):
	# 	print article_text[i], num_list[i]


	target.close()
	# outfile.close()
	# infile.close()
	noise.close()

# getTarget("inputs/all.dat", "img/lg-", "inputs/tbs-30-144*108-color-input.dat", "inputs/tbs-30-144*108-color-targets.dat")
getPixels("all.dat", "img/lg-", "noise.dat")

# getTarget("inputs/all.dat", "img/lg-", "inputs/test-inputs.dat", "inputs/test-targets.dat")


