import cv2
import numpy
from newConx import *
from defClothes import *
import json
from math import *



class ANN(BackpropNetwork):
	"""
	A specialied backprop network for classifying face images for
	the position of head.
	"""
	def classify(self, output):
		"""
		This ensures that that output layer is the correct size for this
		task, and then tests whether the output value is within
		tolerance of 1 (meaning sunglasses are present) or 0 (meaning
		they are not).
		"""
		assert(len(output) == len(article_text))
		maxOutput = max(output)
		return article_text[output.index(maxOutput)]


	def getOutput(self):
		"""
		For the current set of inputs, tests each one in the network to
		determine its classification, compares this classification to
		the targets, and computes the percent correct.
		"""
		if len(self.inputs) == 0:
			print 'no patterns to evaluate'
			return

		for i in range(len(self.inputs)):
			pattern = self.inputs[i]
			output = self.propagate(input=pattern)
			return self.classify(output)


class ClothesAdvisor:
	def __init__(self,  verbose=0):
		self.weights = "ANN/finalWeights"
		self.ANN = self.establishANN()
		self.data = json.load(open('myData.json'))

		self.verbose = verbose
		self.w = 60
		self.h = 80




	def establishANN(self):
		n = ANN()
		n.addLayers(w * h * 4, len(article_text) *2, len(article_text)) 
		# w, h, article_text defined in defClothes.py
		# set the training parameters
		n.setEpsilon(0.3)
		n.setMomentum(0.1)
		n.setReportRate(1)
		n.setTolerance(0.2)
		n.loadWeightsFromFile(self.weights)
		return n

	def distance(self, v1, v2):
		"""
		Returns the Euclidean distance between two vectors.
		"""
		total = 0
		for i in range(len(v1)):
		    total += (v1[i] - v2[i])**2
		return sqrt(total)

	def evaluateClothes(self, imageAddress, retsultFile = "result.json"):
		w = self.w
		h = self.h
		category = self.getClothesCategory(imageAddress)
		if category:
			print category
		else:
			return
		try:
			image_rgb = cv2.imread(imageAddress)
			image_rgb = cv2.resize(image_rgb, (w,h), image_rgb, 0, 0, cv2.INTER_LANCZOS4)
		except:
			print "Image not found, can't get its color"
			return
		colorScheme = self.getColorDistribution(image_rgb, category)
		distance_clusters, in_clusters = self.compareWithClusters(colorScheme, category)
		print distance_clusters, in_clusters
		recommended = self.getClosestCluster(category, distance_clusters)
		evaluation_result = {"category": category, "recommended": recommended, "distance_clusters": distance_clusters, "in_clusters": in_clusters}
		
		with open(retsultFile, 'w') as datafile:
			json.dump(evaluation_result, datafile)
		print evaluation_result
		return evaluation_result

	def getClosestCluster(self, category, distance_clusters):
		min_dis = min(distance_clusters)
		min_index = distance_clusters.index(min_dis)
		in_clusters = [0] * len(article_text)
		in_clusters[min_index] = 1
		return self.getRecommended(category, in_clusters)




	def getRecommended(self, category, in_clusters):
		hist = self.data[category]["hist"]
		image_index_list = []
		for i in range(len(in_clusters)):
			if in_clusters[i]:
				nodes = self.data[category]["hist"][i]["nodes"]
				center = self.data[category]["hist"][i]["center"]
				for j in nodes:
					image_index_list.append(j["imgIndex"])
		
		return {"center": center, "images": image_index_list}




	def compareWithClusters(self, colorScheme, category):
		hist = self.data[category]["hist"]
		vecList = []
		in_clusters = [0] * len(hist)
		for i in range(len(hist)):
			vecList.append(hist[i]["center"])
		minD, label, distance_list = self.minDistInList(colorScheme, vecList)
		for i in range(len(hist)):
			if distance_list[i] <= 0.: #hist[i]["maxDistance"]:
				in_clusters[i] = 1
			else:
				in_clusters[i] = 0

		return distance_list, in_clusters


	def getColorDistribution(self, image, category):
		k_colors = self.data[category]["color"]
		distribution = [0] * len(k_colors)
		for x in range(image.shape[0]):
			for y in range(image.shape[1]):
				rgb = image[x][y]
				minD = 10000
				label = -1
				for c in k_colors:
					d = self.distance(rgb, c)
					if d < minD:
						label = k_colors.index(c)
						minD = d
				distribution[label] += 1
		
		return self.normalize(distribution)


	def minDistInList(self, vec, listVec):
		label = -1
		distance_list = [0] * len(listVec)
		for i in range(len(listVec)):
			d = self.distance(vec, listVec[i])
			distance_list[i] = d
		minD = min(distance_list)
		label = distance_list.index(minD)
		return minD, label, distance_list


	def normalize(self, vec):
		total = sum(vec)
		for i in range(len(vec)):
			vec[i] = vec[i]*1.0/total
		return vec



	def getClothesCategory(self, imageAddress, inputFile = "input/tem.dat"):
		result = self.getInput(imageAddress, inputFile)
		if result:
			self.ANN.loadInputsFromFile(inputFile)
			return self.ANN.getOutput()
		else:
			return result


	def getInput(self, imageAddress, inputFile):
		infile = open(inputFile, "w")
		try:
			display_rgb = cv2.imread(imageAddress)
			image_rgb = numpy.empty((w,h))
			image_rgb = cv2.resize(display_rgb, (w,h), image_rgb, 0, 0, cv2.INTER_LANCZOS4)
		except:
			print "Image not found, can't get its category"
			return 0
		cv2.imshow("c", display_rgb)
		cv2.waitKey(0)
		edges = cv2.Canny(image_rgb, 100, 200)

		for x in range(image_rgb.shape[0]):
			for y in range(image_rgb.shape[1]):
				isEdge = float(edges[x][y])/255.0
				infile.write("%.4f " % isEdge)
				for c in range(image_rgb.shape[2]):
					color = float(image_rgb[x][y][c])/255.0
					infile.write("%.4f " % color)
		infile.close()
		return 1

clothes_list = [517, 1006, 2, 603, 980, 919]
# rec = []
advisor = ClothesAdvisor()

for i in clothes_list:
	result = advisor.evaluateClothes("img/lg-" + str(i) + ".jpg")

	with open("recommended" + str(i) + ".json", 'w') as datafile:
		json.dump(result, datafile)



