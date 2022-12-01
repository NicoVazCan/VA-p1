adjustIntensity:
	python -m src.imgops adjustIntensity

equalizeIntensity:
	python -m src.imgops equalizeIntensity

filterImage:
	python -m src.imgops filterImage

gaussKernel1D:
	python -m src.imgops gaussKernel1D

gaussianFilter:
	python -m src.imgops gaussianFilter

medianFilter:
	python -m src.imgops medianFilter

highBoost:
	python -m src.imgops highBoost

erode:
	python -m src.imgops erode

dilate:
	python -m src.imgops dilate

opening:
	python -m src.imgops opening

closing:
	python -m src.imgops closing

fill:
	python -m src.imgops fill

gradientImage:
	python -m src.imgops gradientImage

edgeCanny:
	python -m src.imgops edgeCanny

cornerHarris:
	python -m src.imgops cornerHarris

all:
	python -m src.imgops		\
		adjustIntensity		\
		equalizeIntensity	\
		filterImage			\
		gaussKernel1D		\
		gaussianFilter		\
		medianFilter		\
		highBoost			\
		erode				\
		dilate				\
		opening				\
		closing				\
		fill				\
		gradientImage		\
		edgeCanny			\
		cornerHarris

clean: cleanFiles
	 rmdir src/imgops/__pycache__
cleanFiles:
	rm src/imgops/__pycache__/*.pyc
	
