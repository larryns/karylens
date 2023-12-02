#!/usr/bin/env python

import argparse
import sys
import typing
import numpy as np
import plotly.express as px

from skimage import io, morphology, color, measure, draw
from skimage.filters import threshold_otsu
from skimage.util import invert

""" Karlens - Find the lengths of individual chromosomes from a karyotype image.
"""


def karylens(inFile: str, maxXLen: int, genSize: int, closing: int) -> typing.List:
	"""
	Compute the lengths of chromosomes given the image in inFile.

	:param inFile: input image file.
	"""
	# Load in the image file
	image = io.imread(inFile)

	# First show the original image
	px.imshow(image).show()

	# Inverting the colours will do better for the skeletonization
	inversion = invert(color.rgb2gray(image))
	px.imshow(inversion).show()

	# Binarize the image. Use Otsu's method to find the threshold between white and
	# black. Then use the threshold to convert the image to just 0's and 1's (False and
	# True).
	threshold = threshold_otsu(inversion)
	binary = inversion > threshold

	# Dilate the image to join small gaps
	solids = morphology.binary_closing(image=binary, footprint=morphology.square(closing))
	px.imshow(solids * 255).show()

	# Now fill the holes
	seed = np.copy(solids)
	seed[1:-1, 1:-1] = True
	filled = morphology.reconstruction(seed, solids, method="erosion")

	# Remove the holes in the image. Reconstruction by erosion specifies that
	px.imshow(filled * 255).show()

	# Now perform the medial axis transform/skeletonization to well, skeletonize the image.
	# MAT is another alternative, but according to the help, skeletonize is better suited when
	# there are less branches. MAT can be run with morphology.medial_axis. I've also considered
	# thinning, but as far as I understand, skeletonize maintains the original size of the object
	# while thinning may or may not. From initial comparisons, the resulting images seem similar
	# regardless of thinning or skeletonizing.
	skeleton = morphology.skeletonize(filled, method='zhang')
	px.imshow(skeleton * 255).show()

	# Get the (segmented) regions.
	labels = measure.label(skeleton, connectivity=2)

	# Go through each region and find the endpoints
	newImage = np.zeros(skeleton. shape, dtype=bool)
	lengths = [ getPaths(skeleton, region.bbox, maxXLen, newImage) for region in measure.regionprops(labels) ]

	# Convert the relative lengths to bp
	minlen = 0
	maxlen = 0
	if len(lengths) % 2 != 0:
	    print("Odd number of chromosomes detected: ", len(lengths), ". Try adjusting the closing distance.", file=sys.stderr)
	    sys.exit(-1)

	for i in range(0, len(lengths), 2):
		# We assume that if the in the pair one length is 
                # significantly shorter than the other then it might be 
                # the sex-chromosomes or a problem, so ignore
		if lengths[i] > 4 * lengths[i+1]:
			lengths[i+1] = lengths[i]
		elif lengths[i+1] > 4 * lengths[i]:
			lengths[i] = lengths[i+1]

		# order the lengths so that the first one is always smaller
		if lengths[i] > lengths[i+1]:
			tmp = lengths[i]
			lengths[i] = lengths[i+1]
			lengths[i] = tmp

		minlen += lengths[i]
		maxlen += lengths[i+1]

	# Now get the genome lengths
	genLengths = [ (int(lengths[i] * genSize / minlen), int(lengths[i] * genSize / maxlen)) for i in range(0, len(lengths), 2) ]
	px.imshow(newImage).show()

	return genLengths

def getLength(x: int, y: int, image: np.array, newImage: np.array, pathLen: int, maxXLen: int, currXLen: int) -> int:
	# First check if there's a pixel at this location or if we've gone too far east
	# Image is column-majored, so y, x NOT x, y
	if not image[y, x]:
		return pathLen

	# If we went too far east, then ignore this path.
	if maxXLen == currXLen:
		return 0

	# Copy current pixel in the path
	newImage[y, x] = True

	# We can only move south-west, south, south-east, and east, but not too far east.
	pathLenSW = getLength(x-1, y+1, image, newImage, pathLen, maxXLen, 0)	# South-west
	pathLenS = getLength(x, y+1, image, newImage, pathLen, maxXLen, 0)		# South
	pathLenSE = getLength(x+1, y+1, image, newImage, pathLen, maxXLen, 0)	# South-east
	pathLenE = getLength(x+1, y, image, newImage, pathLen, maxXLen, currXLen + 1)	# East

	pathLen += max(pathLenE, pathLenS, pathLenSE, pathLenSW)

	return pathLen + 1

def getPaths(image, bbox, maxXLen, newImage):
	minrow, mincol, maxrow, maxcol = bbox

	# Go through the pixels in the bounding box and find the end points.
	lengths = [ getLength(x, y, image, newImage, 0, maxXLen, 0)
				for y in range(minrow, maxrow)
				for x in range(mincol, maxcol)
				if image[y, x] and (not newImage[y, x]) and np.sum(image[(y-1):(y+2), (x-1):(x+2)]) == 2
	]

	return max(lengths)

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Find the lengths of chromosomes in an karyotype image.")
	parser.add_argument(
		'input',
		type=str,
		help='input image'
	)
	parser.add_argument(
		'--maxxlen',
		type=int,
		default=6,
		help='maximum distance between two chromatids to be considered one'
	)
	parser.add_argument(
		"gensize",
		type=int,
		help='size of genome in bp'
	)
	parser.add_argument(
		"--scaffoldfile",
		type=argparse.FileType('r'),
		help='fasta file with the scaffolds'
	)
	parser.add_argument(
		"--closing",
		type=int,
		default=5,
		help='closing size, determines the distance between chromosomes')
	args = parser.parse_args(sys.argv[1:])

	# If there is a scaffold file, we need a gensize
	if args.scaffoldfile is not None and args.gensize is None:
		sys.print("Genome size is required for scaffolds.")
		sys.exit(-1)

	lengths =  karylens(args.input, args.maxxlen, args.gensize, args.closing)

	# check if we have a scaffold file
	if args.scaffoldfile is None:
		print("Min bound\tMax bound\tScaffold Lens")
		for len in lengths:
			print(f"{len[0]}\t{len[1]}")
	else:
		scaffoldLens = []
		currentLen = 0
		for line in args.scaffoldfile:
			if line.startswith('>'):
				scaffoldLens.append(currentLen)
				currentLen = 0
			else:
				currentLen += len(line.strip())
		scaffoldLens.append(currentLen)
		scaffoldLens.sort(reverse=True)

		print("Min bound\tMax bound\tScaffold Lens")
		for len, scafLens in zip(lengths, scaffoldLens):
			print(f"{len[0]}\t{len[1]}\t{scafLens}")
