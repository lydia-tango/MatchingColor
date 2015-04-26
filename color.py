import cv2
import numpy
from defClothes import *

def getPixels(targetFile, imageFilenameRoot, noiseFile):

	target = open(targetFile, "r")



	# outfile = open(outputFile, "w")
	# infile = open(inputFile, "w")
	noise = open(noiseFile, "r")


	# diversity = len(article_text)
	# num_list = [0] * diversity
	noise_list = []

	# max_num = 210

	# verbose = False

	while 1:
		# ignore = False
		# isTop = True 

		line = noise.readline().split()
		if len(line) == 0: break
		noise_list.append(line[0])


	allpixel = None


	for x in range(80):
		# for x in range(1000):
		# ignore = False
		# isTop = True 


		line = target.readline().split()
		if len(line) == 0: break
		
		# if line[1].lower() == "jacket":
		# 	print line[0]
		if line[0] in noise_list:
			# print "noise"
			continue

		if line[1].lower() == "suit":
			print "suits"

			image_rgb = cv2.imread(imageFilenameRoot + line[0]+".jpg")
			h = 40
			w = 30

			image_rgb = cv2.resize(image_rgb, (w,h), image_rgb, 0, 0, cv2.INTER_LANCZOS4)	


			Z = image_rgb.reshape((-1,3))
			# convert to np.float32
			Z = numpy.float32(Z)

			if allpixel == None:
				allpixel = Z
			else:
				allpixel = numpy.concatenate((allpixel, Z))



	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 100
	ret,label,center=cv2.kmeans(allpixel,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)	


	print ret





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


