import numpy
import skimage.io as io
import matplotlib.pyplot as plt
import sys
import os
import cv2
import sys

from src import imgops

imgs = {}

doPlotHist = True

def loadImg(file):
	imgRaw = io.imread(file, as_gray=True)
	
	if imgRaw.dtype == numpy.uint8:
		imgRaw = imgRaw/255.

	return imgRaw

def plotImg(imgIn):
	plt.figure()
	io.imshow(imgIn)
	plt.show(block=False)

def plotHist(imgIn):
	if doPlotHist:
		plt.figure()
		plt.hist(imgIn, bins=256, density=True, histtype='step')
		plt.show(block=False)


def beforeAllTest():
	global imgs
	
	path = "/home/nico/ProyectosGit/VA/p1/test_images"
	files = os.listdir(path)

	for file in files:
		name = os.path.splitext(file)[0]
		relfile = os.path.join(path, file)

		if os.path.isfile(relfile):
			imgs.update({name: relfile})

def afterAllTest():
	plt.show(block=True)

def adjustIntensityTest():
	imgIn = loadImg(imgs["grays"])

	plotImg(imgIn)
	plotHist(imgIn)

	imgOut = imgops.adjustIntensity(imgIn)

	plotImg(imgOut)
	plotHist(imgOut)

	imgOut = imgops.adjustIntensity(imgIn, (0.4,0.6))

	plotImg(imgOut)
	plotHist(imgOut)

	imgOut = imgops.adjustIntensity(imgIn, outRange=(0.45,0.55))

	plotImg(imgOut)
	plotHist(imgOut)

def equalizeIntensityTest():
	imgIn = loadImg(imgs["eq0"])

	plotImg(imgIn)
	plotHist(imgIn)
	
	imgOut = imgops.equalizeIntensity(imgIn)

	plotImg(imgOut)
	plotHist(imgOut)

	imgOut = cv2.equalizeHist(numpy.uint8(imgIn*255))

	plotImg(imgOut)
	plotHist(imgOut)
	"""
	imgOut = imgops.equalizeIntensity(imgIn, nBins=128)

	plotImg(imgOut)
	plotHist(imgOut)
	"""

def filterImageTest():
	kernel = numpy.array([
		[0.05,0.2,0.05],
		[0.2, 0.3,0.2 ],
		[0.05,0.2,0.05]
	])

	imgIn = loadImg(imgs["morph"])

	plotImg(imgIn)

	imgOut = imgops.adjustIntensity(imgops.filterImage(imgIn, kernel))

	plotImg(imgOut)

	imgOut = imgops.adjustIntensity(cv2.filter2D(imgIn, cv2.CV_64F, kernel))

	plotImg(imgOut)

	imgIn = loadImg(imgs["grid"])
	
	plotImg(imgIn)

	imgOut = imgops.adjustIntensity(imgops.filterImage(imgIn, kernel))

	plotImg(imgOut)

	imgOut = imgops.adjustIntensity(cv2.filter2D(imgIn, cv2.CV_64F, kernel))

	plotImg(imgOut)


def gaussKernel1DTest():
	sigma = 1
	kernelIO = imgops.gaussKernel1D(sigma)
	
	ksize = 2*round(3*sigma)+1
	kernelCV = cv2.getGaussianKernel(ksize, sigma).T

	imgIn = loadImg(imgs["morph"])

	plotImg(kernelIO)
	plotImg(kernelCV)

	plotImg(imgIn)

	imgOut = imgops.adjustIntensity(imgops.filterImage(imgIn, kernelIO))

	plotImg(imgOut)

	imgOut = imgops.adjustIntensity(imgops.filterImage(imgIn, kernelCV))

	plotImg(imgOut)

	imgIn = loadImg(imgs["grid"])
	
	plotImg(imgIn)

	imgOut = imgops.adjustIntensity(imgops.filterImage(imgIn, kernelIO))

	plotImg(imgOut)

	imgOut = imgops.adjustIntensity(imgops.filterImage(imgIn, kernelCV))

	plotImg(imgOut)

def gaussianFilterTest():
	imgIn = loadImg(imgs["delta"])

	plotImg(imgIn)
	plotHist(imgIn)

	sigma = 20

	imgOut = imgops.gaussianFilter(imgIn, sigma)

	plotImg(imgOut)
	plotHist(imgOut)

	kernel = imgops.gaussKernel1D(sigma)

	imgOut = imgops.filterImage(imgIn, kernel)
	imgOut = imgops.filterImage(imgOut, kernel.T)

	plotImg(imgOut)
	plotHist(imgOut)

	ksize = 2*round(3*sigma)+1
	ksize = ksize, ksize

	imgOut = cv2.GaussianBlur(
		imgIn, ksize=ksize, sigmaX=sigma, sigmaY=sigma,
		borderType=cv2.BORDER_REFLECT_101
	)

	plotImg(imgOut)
	plotHist(imgOut)

def medianFilterTest():
	ksize = 3

	imgIn = loadImg(imgs["grid"])

	plotImg(imgIn)

	imgOut = imgops.medianFilter(imgIn, ksize)

	plotImg(imgOut)

	imgOut = cv2.medianBlur(numpy.uint8(imgIn*255), ksize)

	plotImg(imgOut)

	ksize = 5

	imgOut = imgops.medianFilter(imgIn, ksize)

	plotImg(imgOut)

	imgOut = cv2.medianBlur(numpy.uint8(imgIn*255), ksize)

	plotImg(imgOut)

	ksize = 7

	imgOut = imgops.medianFilter(imgIn, ksize)

	plotImg(imgOut)

	imgOut = cv2.medianBlur(numpy.uint8(imgIn*255), ksize)

	plotImg(imgOut)

def highBoostTest():
	imgIn = loadImg(imgs["grid"])

	rang = imgIn.max()-imgIn.min()

	plotImg(imgIn)

	A = 2
	method = "median"
	ksize = 9

	imgOut = imgops.highBoost(imgIn, A, method, ksize)

	plotImg(imgOut)

	mask = cv2.medianBlur(numpy.uint8(imgIn*255), ksize)/255
	mask = imgops.adjustIntensity(mask, outRange=(0., rang/2))

	imgOut = imgops.adjustIntensity(A*imgIn - mask)

	plotImg(imgOut)


	A = 2
	method = "gaussian"
	sigma = 1

	imgOut = imgops.highBoost(imgIn, A, method, sigma)

	plotImg(imgOut)

	ksize = 2*round(3*sigma)+1
	ksize = ksize, ksize

	mask = cv2.GaussianBlur(
		imgIn, ksize=ksize, sigmaX=sigma, sigmaY=sigma,
		borderType=cv2.BORDER_REFLECT
	)/255
	mask = imgops.adjustIntensity(mask, outRange=(0., rang/2))

	imgOut = imgops.adjustIntensity(A*imgIn - mask)

	plotImg(imgOut)

def erodeTest():
	imgIn = loadImg(imgs["morph"])
	SE = numpy.array([[1,1,1],[1,1,1],[1,1,1]], dtype=numpy.uint8)
	center = (1,1)

	plotImg(imgIn)

	imgOut = imgops.erode(imgIn, SE, center)

	plotImg(imgOut)

	center = center[::-1]

	imgOut = cv2.erode(imgIn, SE, anchor=center, borderType=cv2.BORDER_REFLECT_101)

	SE = numpy.array([[1,1,1]], dtype=numpy.uint8)
	center = (0,1)

	plotImg(imgIn)

	imgOut = imgops.erode(imgIn, SE, center)

	plotImg(imgOut)

	center = center[::-1]

	imgOut = cv2.erode(imgIn, SE, anchor=center, borderType=cv2.BORDER_REFLECT_101)

	plotImg(imgOut)

def dilateTest():
	imgIn = loadImg(imgs["morph"])
	SE = numpy.array([[1,1,1],[1,1,1],[1,1,1]], dtype=numpy.uint8)
	center = (1,1)

	plotImg(imgIn)

	imgOut = imgops.dilate(imgIn, SE, center)

	plotImg(imgOut)

	center = center[::-1]

	imgOut = cv2.dilate(imgIn, SE, anchor=center, borderType=cv2.BORDER_REFLECT_101)

	plotImg(imgOut)

	SE = numpy.array([[1,1,1]], dtype=numpy.uint8)
	center = (0,1)

	plotImg(imgIn)

	imgOut = imgops.dilate(imgIn, SE, center)

	plotImg(imgOut)

	center = center[::-1]

	imgOut = cv2.dilate(imgIn, SE, anchor=center, borderType=cv2.BORDER_REFLECT_101)

	plotImg(imgOut)

def openingTest():
	imgIn = loadImg(imgs["morph1"])
	SE = numpy.array([[0,1,0],[1,1,1],[0,1,0]], dtype=numpy.uint8)
	center = (1,1)

	plotImg(imgIn)

	imgOut = imgops.opening(imgIn, SE, center)

	plotImg(imgOut)

	center = center[::-1]

	imgOut = cv2.morphologyEx(
		numpy.uint8(imgIn*255), cv2.MORPH_OPEN, SE,
		anchor=center, iterations=1, borderType=cv2.BORDER_REFLECT_101
	)

	plotImg(imgOut)

def closingTest():
	imgIn = loadImg(imgs["morph1"])
	SE = numpy.array([[0,1,0],[1,1,1],[0,1,0]], dtype=numpy.uint8)
	center = (1,1)

	plotImg(imgIn)

	imgOut = imgops.closing(imgIn, SE, center)

	plotImg(imgOut)

	center = center[::-1]

	imgOut = cv2.morphologyEx(
		numpy.uint8(imgIn*255), cv2.MORPH_CLOSE, SE,
		anchor=center, iterations=1, borderType=cv2.BORDER_REFLECT_101
	)

	plotImg(imgOut)

def fillTest():
	imgIn = loadImg(imgs["fill"])
	seeds = [(16,16), (11,3)]
	SE = numpy.array([[0,1,0],[1,1,1],[0,1,0]], dtype=numpy.uint8)
	center = (1,1)

	plotImg(imgIn)

	imgOut = imgops.fill(imgIn, seeds, SE, center)

	plotImg(imgOut)

def gradientImageTest():
	imgIn = loadImg(imgs["cuadrado"])

	plotImg(imgIn)

	gy, gx = imgops.gradientImage(imgIn, "Roberts")

	plotImg(imgops.adjustIntensity(gy))
	plotImg(imgops.adjustIntensity(gx))
	plotImg(imgops.adjustIntensity(abs(gy)+abs(gx)))

	gy = cv2.filter2D(imgIn, cv2.CV_64F, imgops.gradKernels["Roberts"][0])
	gx = cv2.filter2D(imgIn, cv2.CV_64F, imgops.gradKernels["Roberts"][1])

	plotImg(imgops.adjustIntensity(gy))
	plotImg(imgops.adjustIntensity(gx))
	plotImg(imgops.adjustIntensity(abs(gy)+abs(gx)))


	gy, gx = imgops.gradientImage(imgIn, "CentralDiff")

	plotImg(imgops.adjustIntensity(gy))
	plotImg(imgops.adjustIntensity(gx))
	plotImg(imgops.adjustIntensity(abs(gy)+abs(gx)))

	gy = cv2.filter2D(imgIn, cv2.CV_64F, imgops.gradKernels["CentralDiff"][0])
	gx = cv2.filter2D(imgIn, cv2.CV_64F, imgops.gradKernels["CentralDiff"][1])

	plotImg(imgops.adjustIntensity(gy))
	plotImg(imgops.adjustIntensity(gx))
	plotImg(imgops.adjustIntensity(abs(gy)+abs(gx)))


	gy, gx = imgops.gradientImage(imgIn, "Prewitt")

	plotImg(imgops.adjustIntensity(gy))
	plotImg(imgops.adjustIntensity(gx))
	plotImg(imgops.adjustIntensity(abs(gy)+abs(gx)))

	gy = cv2.filter2D(imgIn, cv2.CV_64F, imgops.gradKernels["Prewitt"][0])
	gx = cv2.filter2D(imgIn, cv2.CV_64F, imgops.gradKernels["Prewitt"][1])

	plotImg(imgops.adjustIntensity(gy))
	plotImg(imgops.adjustIntensity(gx))
	plotImg(imgops.adjustIntensity(abs(gy)+abs(gx)))


	gy, gx = imgops.gradientImage(imgIn, "Sobel")

	plotImg(imgops.adjustIntensity(gy))
	plotImg(imgops.adjustIntensity(gx))
	plotImg(imgops.adjustIntensity(abs(gy)+abs(gx)))


	gx = cv2.Sobel(
		imgIn, cv2.CV_64F, 1, 0, ksize=3,
		borderType=cv2.BORDER_REFLECT_101
	)

	gy = cv2.Sobel(
		imgIn, cv2.CV_64F, 0, 1, ksize=3,
		borderType=cv2.BORDER_REFLECT_101
	)

	plotImg(imgops.adjustIntensity(gy))
	plotImg(imgops.adjustIntensity(gx))
	plotImg(imgops.adjustIntensity(abs(gy)+abs(gx)))

def edgeCannyTest():
	imgIn = loadImg(imgs["image5"])

	plotImg(imgIn)

	imgOut = imgops.edgeCanny(imgIn, 2, 0.01, 0.1)

	plotImg(imgOut)

	imgOut = imgops.edgeCanny(imgIn, 2, 0.1, 0.2)

	plotImg(imgOut)

	imgOut = imgops.edgeCanny(imgIn, 2, 0.2, 0.3)

	plotImg(imgOut)

	imgOut = imgops.edgeCanny(imgIn, 6, 0.01, 0.05)

	plotImg(imgOut)

	imgOut = imgops.edgeCanny(imgIn, 6, 0.05, 0.1)

	plotImg(imgOut)

	imgOut = imgops.edgeCanny(imgIn, 6, 0.1, 0.2)

	plotImg(imgOut)

	"""
	imgOut = cv2.Canny(numpy.uint8(imgIn*255), 0.9, 100, 200)

	plotImg(imgOut)
	"""


def putCorners(imgIn, imgCorner):
	SE = numpy.array([
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[1,1,1,1,1,1,1],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
	], dtype=numpy.uint8)
	center = (3,3)

	imgOut = numpy.uint8(numpy.dstack((imgIn, imgIn, imgIn))*255)
	corners = imgops.dilate(imgCorner > 0, SE, center)
	imgOut[corners, :] = [255, 0, 0]

	return imgOut

def cornerHarrisTest():
	imgIn = loadImg(imgs["circles"])

	plotImg(imgIn)

	outCorners, harrisMap = imgops.cornerHarris(imgIn, 0.6, 0.2, 0.001)

	imgOut = putCorners(imgIn, outCorners)

	plotImg(imgops.adjustIntensity(harrisMap))
	plotImg(outCorners)
	plotImg(imgOut)


tests = {
	"adjustIntensity":		adjustIntensityTest,
	"equalizeIntensity":	equalizeIntensityTest,
	"filterImage":			filterImageTest,
	"gaussKernel1D":		gaussKernel1DTest,
	"gaussianFilter":		gaussianFilterTest,
	"medianFilter":			medianFilterTest,
	"highBoost":			highBoostTest,
	"erode":				erodeTest,
	"dilate":				dilateTest,
	"opening":				openingTest,
	"closing":				closingTest,
	"fill":					fillTest,
	"gradientImage":		gradientImageTest,
	"edgeCanny":			edgeCannyTest,
	"cornerHarris":			cornerHarrisTest
}

def main(argc, argv):
	if argc == 1:
		print("Uso: imgops FUNC...")
		print("FUNC:")
		for test,_ in tests.items():
			print("\t{}".format(test)) 
	else:
		beforeAllTest()

		for test in argv[1:]:
			if test in tests:
				tests[test]()
			else:
				print("TEST ({}) no reconocido y saltado".format(test))
				
		afterAllTest()


if __name__ == "__main__":
	main(len(sys.argv), sys.argv)
