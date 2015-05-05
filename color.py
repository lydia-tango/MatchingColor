import cv2
import numpy
import json
# where we define the category of men clothes
from defClothes import *
from matplotlib import pyplot as plt
from math import *


class colorSchemeCategorizer:
	def __init__(self, article_text, article_list, noiseFile = "noise.dat", verbose=0):
		self.data  = {}
		self.noise_list = []
		self.article_text = article_text
		self.article_list = article_list
		self.initializeData()
		self.getNoiseList(noiseFile)
		self.K = 25
		self.P = 10


	def initializeData(self):
		self.data = dict()
		for text in self.article_text:
			self.data[text] = {"color": [], "hist": []}
		print "initialized data:", self.data

	def getNoiseList(self, noiseFile):
		# noiseFile stores all the index of noise self.data 
		noise = open(noiseFile, "r")
		# add all indeces of noices into noise_list
		while 1:
			line = noise.readline().split()
			if len(line) == 0: break
			self.noise_list.append(line[0])
		noise.close()

	def distance(self, v1, v2):
		"""
		Returns the Euclidean distance between two vectors.
		"""
		total = 0
		for i in range(len(v1)):
		    total += (v1[i] - v2[i])**2
		return sqrt(total)

	def getClusters(self, targetFile = "rawData.dat", imageFilenameRoot = "img/lg-"):
		article_text = self.article_text
		article_list = self.article_list
		diversity = len(article_text)
		# num_list = [0] * diversity



		for article_index in range(diversity):
			image_index = []

			num_clothes = 0
			category_text = article_text[article_index]

			# targetFile stores all the category information of the images
			target = open(targetFile, "r")
			allpixel = numpy.empty((1,1,1), 'uint8')	# all pixel is a n * 3 matrix of all n pixels in images

			# loop through images 
			for i in range(1000):
			# while 1:	
				line = target.readline().split()
				if len(line) == 0: break
				
				# ignore noise images
				if line[0] in self.noise_list:
					# print "noise"
					continue

				if line[1].lower() in article_list[article_index]:
					if num_clothes >= 50:
						continue
					
					try:
						image_rgb = cv2.imread(imageFilenameRoot + line[0]+".jpg")
						image_rgb = cv2.resize(image_rgb, (w,h), image_rgb, 0, 0, cv2.INTER_LANCZOS4)	
					except:
						continue
					num_clothes += 1
					image_index.append((line[0],category_text))
					print imageFilenameRoot + line[0]+".jpg", category_text


					# construct a wh * 3 matrix of all pixels in the image
					image_pixels = image_rgb.reshape((-1,3))
					# convert to np.float32
					image_pixels = numpy.float32(image_pixels)
					# append the image_pixels to a large matrix
					if allpixel.size < 4:
						allpixel = image_pixels
					else:
						allpixel = numpy.concatenate((allpixel, image_pixels))

			target.close()
			print "Loaded", num_clothes, article_text[article_index]
			# define criteria, number of clusters(K) and apply kmeans()
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			ret,label,center = cv2.kmeans(allpixel, self.K,criteria,10, cv2.KMEANS_RANDOM_CENTERS)	
			self.data[category_text]["color"] = center.tolist()
			scale = 30
			center_color_display = numpy.empty((scale, self.K * scale, 3), 'uint8')
			for i, c in enumerate(center):
				center_color_display[:, i*scale: (i+1)*scale] = c
			cv2.imwrite('analysis/' + article_text[article_index] + str(self.K) + '.jpg', center_color_display)

			print "Finished kmeans round 1 with", num_clothes, article_text[article_index]


			label_each_image_list = numpy.vsplit(label, num_clothes)
			allHist = numpy.empty((1,1,1), 'float32')
			for label_each_image in label_each_image_list:
				center = numpy.uint8(center)
				res = center[label_each_image.flatten()]
				res2 = res.reshape(h, w, 3)
				# get a histogram of color distribution of each image
				hist, bins = numpy.histogram(label_each_image, self.K, [0, self.K])
				hist = hist.astype("f")
				hist /= h*w # normalize distribution

			
				if allHist.size < 4:
					allHist = hist
				else:
					allHist = numpy.vstack((allHist, hist))
			
			P = 5
			# P = num_clothes/2
			# while P > 10:
			# 	P = P/2

			histRet,histLabel,histCenterList=cv2.kmeans(allHist,P,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

			print category_text
			print self.data
			for histCenter in histCenterList:
				self.data[category_text]["hist"].append({"center": histCenter.tolist(), "nodes": [], "maxDistance": 0})
			for i in range(num_clothes):
				center_index = histLabel[i][0]
				hist_i = allHist[i].tolist()
				mean_i = histCenterList[center_index].tolist()

				dist = self.distance(hist_i, mean_i)

				self.data[category_text]["hist"][center_index]["nodes"].append({"imgIndex": image_index[i][0], "category": image_index[i][1], "distance": dist, "distribution": hist_i})
				# print dist
			print self.data

	def getMaxDistance(self):
		for k, v in self.data.iteritems():
			hist_list = self.data[k]["hist"]
			
			for i, hist in enumerate(hist_list):
				dis_list = []
				for node in hist["nodes"]:
					dis_list.append(node["distance"])
				self.data[k]["hist"][i]["maxDistance"] = max(dis_list)



	def writeJson(self, dFile = "myData.json"):
		self.getMaxDistance()
		with open(dFile, 'w') as datafile:
			json.dump(self.data, datafile)

	def writeVisJson(self, dFile = "visualization.json"):
		data = {"name": "root", "children": []}

		# second layer: category
		for a in range(len(article_text)):
			cat = article_text[a]
			myCat = self.data[cat]
			cat_data = {"name": cat, "color": myCat["color"] ,"children": []}
			# third later clusters
			for c in range(len(myCat["hist"])):
				myCluster = myCat["hist"][c]
				center = myCat["hist"][c]["center"]

				cluster_data = {"name": cat + " color scheme " + str(c), "color": myCat["color"], "distribution": center,"children": []}
				# fourth layer images
				for i in range(len(myCluster["nodes"])):
					myNode = myCluster["nodes"][i]
					color = myCat["color"]
					cluster_data["children"].append({"name" : myNode["imgIndex"], "size": 1, "distance": myNode["distance"], "category": myNode["category"], "distribution": myNode["distribution"], "color": color})

				cat_data["children"].append(cluster_data)

			data["children"].append(cat_data)
		with open(dFile, 'w') as datafile:
			json.dump(data, datafile)



	

c = colorSchemeCategorizer(article_text, article_list)
c.getClusters()
c.writeJson()
c.writeVisJson()



