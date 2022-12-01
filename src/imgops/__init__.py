import numpy
from typing import List, Dict, Tuple, Union, Callable

"""
DESCRIPTION
	Libreria con las funciones de la primera practica.
AUTHOR
	Nicolas Vazquez Cancela.
"""

# Operaciones sobre historgramas:

def adjustIntensity(inImage: numpy.ndarray,#[float]
                    inRange: Tuple[float, float]=(),
                    outRange: Tuple[float, float]=(0., 1.)) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		inRange: Vector 1x2 con el rango de niveles de intensidad [imin, imax]
			de entrada. Si el vector está vacı́o (por defecto), el mı́nimo y máximo
			de la imagen de entrada se usan como imin e imax.
		outRange: Vector 1x2 con el rango de niveles de instensidad [omin, omax]
			de salida. El valor por defecto es [0 1].
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""
	if not inRange:
		inRange = numpy.min(inImage), numpy.max(inImage)

	if inRange == outRange:
		return inImage

	return (inImage - inRange[0]) * ((outRange[1]-outRange[0])/
									(inRange[1]-inRange[0])) + outRange[0]

def equalizeIntensity(inImage: numpy.ndarray,
                      nBins: int=256) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		nBins: Número de bins utilizados en el procesamiento. Se asume que el
			intervalo de entrada [0 1] se divide en nBins intervalos iguales para
			hacer el procesamiento, y que la imagen de salida vuelve a quedar en
			el intervalo [0 1]. Por defecto 256.
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""

	pixArray = numpy.uint8(inImage.flatten()*(nBins-1))

	hist = numpy.zeros(nBins)

	for p in pixArray:
		hist[p] += 1

	cdf = numpy.cumsum(hist)

	h = ((cdf - cdf.min())/
	     (inImage.shape[0]*inImage.shape[1]-cdf.min())
	)

	return h[pixArray].reshape(inImage.shape)
	

# Filtros:

def convImage(inImage: numpy.ndarray,#[float]
                clipShape: Tuple[int, int],
                convFunc: Callable[[numpy.ndarray], numpy.float64]
                ) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		clipShape: Tamano del area a de convolucion en la imagen
		convFunc: Funcion que se aplica a cada recorte de la imagen
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""

	if len(clipShape) != 2:
		clipShape = (clipShape[0], 1)

	assert clipShape[0]%2 and clipShape[1]%2

	mv = clipShape[0]//2
	mh = clipShape[1]//2

	outImage = numpy.ndarray(inImage.shape, dtype=inImage.dtype)

	for y in range(inImage.shape[0]):
		indexV = [[
				-i if i<0 else
				inImage.shape[0]-i-2 if i>=inImage.shape[0] else
				i
			]
			for i in range(y-mv,y+mv+1)
		]

		for x in range(inImage.shape[1]):
			indexH = [(
					-i if i<0 else
					inImage.shape[1]-i-2 if i>=inImage.shape[1] else
					i
				)
				for i in range(x-mh,x+mh+1)
			]

			outImage[y, x] = convFunc(inImage[indexV,indexH])
	
	return outImage

def filterImage(inImage: numpy.ndarray,#[float]
                kernel: numpy.ndarray#[float]
                ) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		kernel: Matriz PxQ con el kernel del filtro de entrada. Se asume que la
			posición central del filtro está en (bP/2c + 1, bQ/2c + 1).
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""

	return convImage(inImage, kernel.shape, lambda clip: numpy.sum(clip*kernel))

def gaussKernel1D(sigma: float) -> numpy.ndarray:#[float]
	"""
	INPUT
		sigma: Parámetro σ de entrada.
	OUTPUT
		kernel: Vector 1xN con el kernel de salida, teniendo en cuenta que:
			· El centro x = 0 de la Gaussiana está en la posición bN/2c + 1.
			· N se calcula a partir de σ como N = 2*round(3*sigma)+1.
	"""

	mk = round(3*sigma)
	
	index = numpy.array([range(-mk,mk+1)])
	return (numpy.exp(-(index**2 / (2 * sigma**2)))/
	        (numpy.sqrt(2*numpy.pi) * sigma)
	)

def gaussianFilter(inImage: numpy.ndarray,#[float]
                   sigma: float) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		sigma: Parámetro σ de entrada.
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""

	mk = round(3*sigma)

	index = numpy.ndarray((mk*2+1, mk*2+1))

	index1D = numpy.arange(-mk, mk+1,1)

	for i in index1D+mk:
		index[:,i] = index1D

	kernel = (numpy.exp(-(index**2 + numpy.transpose(index)**2)/(2*sigma**2))/
	          (2*numpy.pi*sigma**2)
	)

	return filterImage(inImage, kernel)

def medianFilter(inImage: numpy.ndarray,#[float]
                 filterSize: int) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		filterSice: Valor entero N indicando que el tamaño de ventana es de NxN.
			La posición central de la ventana es (bN/2c + 1, bN/2c + 1).
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""

	assert filterSize%2

	return convImage(inImage,
		(filterSize, filterSize),
		lambda clip: numpy.sort(clip.flatten())[(filterSize + 1)*(filterSize//2)])



def highBoost(inImage: numpy.ndarray,#[float]
              A: float, method: str,
              param:Union[int, float]) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		A: Factor de amplificación del filtro high-boost.
		method: Método de suavizado. Los valores pueden ser:
			· ’gaussian’, indicando que usará la función gaussianFilter.
			· ’median’, indicando que se usará la función medianFilter.
		param: Valor del parámetro del filtro de suavizado. Proporcionará el valor
			de σ en el caso del filtro Gaussiano, y el tamaño de ventana en el caso
			del filtro de medianas.
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""

	rang = inImage.max()-inImage.min()
	
	if method == "gaussian":
		imgInvSmoth = adjustIntensity(-gaussianFilter(inImage, param), outRange=(0., rang/2))
	elif method == "median":
		imgInvSmoth = adjustIntensity(-medianFilter(inImage, param), outRange=(0., rang/2))
	else:
		raise Exception("metodo {} no reconocido".format(method))

	return adjustIntensity(A*inImage + imgInvSmoth)

def erode(inImage: numpy.ndarray,#[float]
          SE: numpy.ndarray,#[int]
          center :Tuple[int, int]=()) -> numpy.ndarray:#[float] 
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
		center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el
			[0 0] es la esquina superior izquierda. Si es un vector vacı́o (valor por
			defecto), el centro se calcula como (bP/2c + 1, bQ/2c + 1).
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""

	if not center:
		center = (SE.shape[0]//2, SE.shape[1]//2)

	assert center[0] < SE.shape[0] and center[1] < SE.shape[1]
	
	index = numpy.nonzero(SE)
	index = index[0]+SE.shape[0]-1-center[0], index[1]+SE.shape[1]-1-center[1]

	return convImage(inImage, (SE.shape[0]*2-1, SE.shape[1]*2-1), lambda clip: numpy.min(clip[index]))

def dilate(inImage: numpy.ndarray,#[float]
           SE: numpy.ndarray,#[uint8]
           center :Tuple[int, int]=()) -> numpy.ndarray:#[float] 
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
		center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el
			[0 0] es la esquina superior izquierda. Si es un vector vacı́o (valor por
			defecto), el centro se calcula como (bP/2c + 1, bQ/2c + 1).
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""

	if not center:
		center = (SE.shape[0]//2, SE.shape[1]//2)

	assert center[0] < SE.shape[0] and center[1] < SE.shape[1]
	
	index = numpy.nonzero(SE)
	index = index[0]+SE.shape[0]-1-center[0], index[1]+SE.shape[1]-1-center[1]

	return convImage(inImage, (SE.shape[0]*2-1, SE.shape[1]*2-1), lambda clip: numpy.max(clip[index]))

def opening(inImage: numpy.ndarray,#[float]
            SE: numpy.ndarray,#[int]
            center :Tuple[int, int]=()) -> numpy.ndarray:#[float] 
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
		center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el
			[0 0] es la esquina superior izquierda. Si es un vector vacı́o (valor por
			defecto), el centro se calcula como (bP/2c + 1, bQ/2c + 1).
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""
	return dilate(erode(inImage, SE, center), SE, center)

def closing(inImage: numpy.ndarray,#[float]
            SE: numpy.ndarray,#[int]
            center :Tuple[int, int]=()) -> numpy.ndarray:#[float] 
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
		center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el
			[0 0] es la esquina superior izquierda. Si es un vector vacı́o (valor por
			defecto), el centro se calcula como (bP/2c + 1, bQ/2c + 1).
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""
	return erode(dilate(inImage, SE, center), SE, center)

def fill(inImage: numpy.ndarray,#[float]
	     seeds: numpy.ndarray,#[int]
         SE: numpy.ndarray=None,#[int]
         center :Tuple[int, int]=()) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		seeds: Matriz Nx2 con N coordenadas (fila,columna) de los puntos semilla.
		SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante de
			conectividad. Si es un vector vacı́o se asume conectividad 4 (cruz 3×3).
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""
	if SE is None:
		SE = numpy.array([[0,1,0],[1,1,1],[0,1,0]], dtype=numpy.uint8)

	if not center:
		center = (SE.shape[0]//2, SE.shape[1]//2)

	assert center[0] < SE.shape[0] and center[1] < SE.shape[1]

	mask0 = numpy.zeros(inImage.shape, dtype=numpy.float64)
	
	for seed in seeds:
		mask1 = mask0.copy()
		mask1[seed] = inImage[seed]

		while numpy.any(mask0 != mask1):
			mask0 = mask1.copy()

			mask1 = inImage * dilate(mask1, SE, center)

	return mask1

gradKernels = {
	"Roberts"     : (numpy.array([[0,1,0],[-1,0,0],[0,0,0]], dtype=numpy.int8),
					 numpy.array([[-1,0,0],[0,1,0],[0,0,0]], dtype=numpy.int8)),

	"CentralDiff" : (numpy.array([[-1],[0],[1]], dtype=numpy.int8),
					 numpy.array([[-1,0,1]],       dtype=numpy.int8)),

	"Prewitt"     : (numpy.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=numpy.int8),
					 numpy.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=numpy.int8)),

	"Sobel"       : (numpy.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=numpy.int8),
					 numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=numpy.int8))

}

def gradientImage(inImage: numpy.ndarray,#[float]
                  operator: str) -> Tuple[int, int]:
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		operator: Permite seleccionar el operador utilizado mediante los valores:
			’Roberts’, ’CentralDiff’, ’Prewitt’ o ’Sobel’.
	OUTPUT
		gx, gy: Componentes Gx y Gy del gradiente.
	"""
	kernels = gradKernels[operator]

	return filterImage(inImage, kernels[0]), filterImage(inImage, kernels[1])

getAngle = numpy.vectorize(lambda y, x:
			numpy.arctan(y/x) if x != 0
			else numpy.sign(y)*numpy.pi/2 if y != 0
			else numpy.pi
		)

def getNexPix(pix, ang):
	if ang < 0:
		ang += numpy.pi*2
	n = (ang+numpy.pi/8)//(2*numpy.pi/8)

	return  ((pix[0],  pix[1]+1) if n == 0
		else (pix[0]+1,pix[1]+1) if n == 1
		else (pix[0]+1,pix[1]  ) if n == 2
		else (pix[0]+1,pix[1]-1) if n == 3
		else (pix[0],  pix[1]-1) if n == 4
		else (pix[0]-1,pix[1]-1) if n == 5
		else (pix[0]-1,pix[1]  ) if n == 6
		else (pix[0]-1,pix[1]+1)
	)

def angSupressNonMax(angImg, magImg):
	
	assert angImg.shape == magImg.shape

	supMaxImg = numpy.zeros(angImg.shape, dtype=numpy.float64)

	for y in range(angImg.shape[0]):
		for x in range(angImg.shape[1]):
			if magImg[y,x]:
				fpix = getNexPix((y,x), angImg[y,x])
				bpix = getNexPix((y,x), angImg[y,x]+numpy.pi)

				isfin = ((0 <= fpix[0] < angImg.shape[0]) and
				         (0 <= fpix[1] < angImg.shape[1]))

				isbin = ((0 <= bpix[0] < angImg.shape[0]) and
				         (0 <= bpix[1] < angImg.shape[1]))

				if not isfin and isbin:
					if magImg[y,x] > magImg[bpix]:
						supMaxImg[y,x] = magImg[y,x]

				elif isfin and not isbin:
					if magImg[y,x] > magImg[fpix]:
						supMaxImg[y,x] = magImg[y,x]

				elif isfin and isbin:
					if (magImg[y,x] >= magImg[fpix] and
						magImg[y,x] >= magImg[bpix]):

						supMaxImg[y,x] = magImg[y,x]
				else:
					supMaxImg[y,x] = magImg[y,x]

	return supMaxImg


def recHisteresis(histImg, supMaxImg, angImg, pix, tlow, thigh):
	if histImg[pix] != -1:
		return
	if supMaxImg[pix] < tlow:
		histImg[pix] = 0
		return

	histImg[pix] = thigh

	leftPix  = getNexPix(pix, angImg[pix]+numpy.pi/2)
	rightPix = getNexPix(pix, angImg[pix]-numpy.pi/2)

	if (0 <= leftPix[0] < histImg.shape[0] and 
		0 <= leftPix[1] < histImg.shape[1]):
		recHisteresis(histImg, supMaxImg, angImg, leftPix, tlow, thigh)

	if (0 <= rightPix[0] < histImg.shape[0] and 
		0 <= rightPix[1] < histImg.shape[1]):
		recHisteresis(histImg, supMaxImg, angImg, rightPix, tlow, thigh)

def histeresis(angImg, supMaxImg, tlow, thigh):
	histImg = numpy.full(supMaxImg.shape, -1., dtype=numpy.float64)

	for y in range(supMaxImg.shape[0]):
		for x in range(supMaxImg.shape[1]):

			if histImg[y,x] == -1:
				if supMaxImg[y,x] >= thigh:
					recHisteresis(histImg, supMaxImg, angImg, (y,x), tlow, thigh)
				elif supMaxImg[y,x] < tlow:
					histImg[y,x] = 0

	histImg[histImg == -1] = 0

	return histImg

def edgeCanny(inImage: numpy.ndarray,#[float]
              sigma: float, tlow: float, thigh: float) -> numpy.ndarray:#[float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		sigma: Parámetro σ del filtro Gaussiano.
		tlow, thigh: Umbrales de histéresis bajo y alto, respectivamente.
	OUTPUT
		outImage: Matriz MxN con la imagen de salida.
	"""
	
	assert tlow < thigh

	angIt = numpy.pi/(1+3)

	smothImg = gaussianFilter(inImage, sigma)

	gy, gx = gradientImage(smothImg, "Sobel")

	rawAngImg = getAngle(gy, gx)

	angImg = ((rawAngImg+angIt/2)//angIt)*angIt
	magImg = numpy.sqrt(pow(gy,2)+pow(gx,2))

	supMaxImg = angSupressNonMax(angImg, magImg)

	histImg = histeresis(angImg, supMaxImg, tlow, thigh)

	return adjustIntensity(histImg)


# Opcional

def windowSupressNonMax(clip):
	centro = clip[1,1]
	vecinos = clip[[0,0,0,1,1,2,2,2],[0,1,2,0,2,0,1,2]]
	
	return centro if numpy.all(vecinos < centro) else 0


def cornerHarris(inImage: numpy.ndarray,#[float]
                 sigmaD: float, sigmaI: float,
                 t: float) -> Tuple[numpy.ndarray, numpy.ndarray]:#[float], [float]
	"""
	INPUT
		inImage: Matriz MxN con la imagen de entrada.
		sigmaD: Escala de diferenciación Gaussiana
		sigmaI: Escala de Integración Gaussiana
		t: Umbral de detección de esquinas
	OUTPUT
		outCorners, harrisMap: La función devolverá el cálculo de la métrica de
			Harris para cada punto en harrisMap, y el mapa tras supresión no
			máxima y umbralización en outCorners.
	"""
	shapeN = (3,3)
	k = 0.04
	gy, gx = gradientImage(inImage, "Sobel")
	mv = shapeN[0]//2
	mh = shapeN[1]//2

	Iyy = gaussianFilter(gy**2, sigmaD)
	Ixx = gaussianFilter(gx**2, sigmaD)
	Iyx = gaussianFilter(gy*gx, sigmaD)

	iIyy = sigmaI*gaussianFilter(Iyy, sigmaI)
	iIxx = sigmaI*gaussianFilter(Ixx, sigmaI)
	iIyx = sigmaI*gaussianFilter(Iyx, sigmaI)

	harrisMap = numpy.ndarray(inImage.shape, dtype=inImage.dtype)

	for y in range(inImage.shape[0]):
		for x in range(inImage.shape[1]):
			M = numpy.array([
				[iIxx[y,x], iIyx[y,x]],
				[iIyx[y,x], iIyy[y,x]]
			])

			harrisMap[y,x] = numpy.linalg.det(M) - k*numpy.trace(M)**2

	SE = numpy.ones((3,3))
	center = (1,1)

	outCorners = convImage(harrisMap, (3,3), windowSupressNonMax) > t

	return outCorners, harrisMap

