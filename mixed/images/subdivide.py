#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
from PIL import ImageOps
from PIL import ImageChops
from PIL import ImageStat

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

from os.path import isfile

import ImageUtils

from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# http://pybrain.org/docs/api/structure/modules.html
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import SigmoidLayer
from pybrain.structure.modules   import TanhLayer

from pybrain.utilities           import percentError
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader


# image parameters
# chunk width and height calculated to split a 1280x720 image in a 20x20 grid
chunkWidth = 64
chunkHeight = 36
chunkSize = chunkWidth * chunkHeight

# filesystem parameters
basePath = os.path.dirname(os.path.realpath(__file__))
imagePath = basePath + '/subdivide'

sampleDataOkPath = imagePath + '/ok'
sampleDataKoPath = imagePath + '/ko'
testDataPath = imagePath + '/test'

chunkDifferencesPath = imagePath + '/chunks_differences'

XmlNetworkName = "subdivideTanHSigmoidBias.xml"

def getImageAndMaskDifferenceChunks(imgPath):
    maskPath = imgPath.replace('.jpg','.png')
    srcImage = Image.open(imgPath)
    maskImage = Image.open(maskPath)
    imageChunks = ImageUtils.splitImageInChunks(ImageUtils.getBlackAndWhite(srcImage), chunkWidth, chunkHeight)
    maskChunks = ImageUtils.splitImageInChunks(ImageUtils.getPNGBWMask(maskImage), chunkWidth, chunkHeight)

    pairs = zip(imageChunks, maskChunks)
    # print "arrays: %s, %s" % (len(imageChunks), len(maskChunks))
    chunks = []
    for p in pairs:
        # immagine - maschera
        differenceChunk = ImageChops.difference(p[0],p[1])

        # maschera - immagine
        # differenceChunk = ImageChops.difference(p[1],p[0])
        chunks.append(differenceChunk)
        # print ImageStat.Stat(differenceChunk).mean
        # print "modes: %s , %s " % (p[0].mode, p[1].mode)
        # print "jpg chunk: %sx%s" % (p[0].size)
        # print "png chunk: %sx%s" % (p[1].size)

    return chunks

# 253.5
# discard after threshold if mean > threshold
# discard before threshold if mean < threshold

def chunkIsRelevant(c, threshold=255):
    stats = ImageStat.Stat(c)
    # la mediana mi indica qual'e' il valore (colore) piu frequente
    # la media del colore mi indica quanto è bianca/nera (perhe tratto solo pixel bianchi o neri)
    print " mean %s, median %s" % ( stats.mean, stats.median)

    #if stats.mean[0] > threshold:
    if stats.mean[0] < threshold:
        print "mean at %s, ignoring chunk" % (stats.mean)
        return False

    return True

def saveChunks(inputDir, outputDir, relevanceThreshold=255):
    print "saveChunks from %s to %s " % (inputDir, outputDir)
    imageIndex = 1
    for fileName in os.listdir(inputDir):
        path = inputDir + '/' + fileName
        f, file_extension = os.path.splitext(path)

        if file_extension == '.png':
            chunks = ImageUtils.splitImageInChunks(ImageUtils.getPNGBWMask(Image.open(path)), chunkWidth, chunkHeight)
            for index, c in enumerate(chunks):
                if(chunkIsRelevant(c)):
                    c.save("%s/chunk-%s-%s-%s%s" % (outputDir, imageIndex, int(index % 20), int(index / 20), file_extension))
            imageIndex+=1
            continue

        if not isfile(path) or file_extension != '.jpg':
            continue


        print "- opening "+path
        # chunks = getImageAndMaskDifferenceChunks(path)

        srcImage = Image.open(path)
        if file_extension == '.jpg':
            chunks = ImageUtils.splitImageInChunks(ImageUtils.getBlackAndWhite(srcImage), chunkWidth, chunkHeight)

        # else:
        #     chunks = ImageUtils.splitImageInChunks(ImageUtils.getPNGBWMask(srcImage), chunkWidth, chunkHeight)


        for index, c in enumerate(chunks):
            savePath = "%s/chunk-%s-%s-%s%s" % (outputDir, imageIndex, int(index % 20), int(index / 20), file_extension)
            if(chunkIsRelevant(c, relevanceThreshold)):
                c.save(savePath)

    print "savechunks - finished"



def saveDifferenceChunks(inputDir, outputDir, alsoSaveChunks=False, relevanceThreshold=255):
    print "saveDifferenceChunks from %s to %s " % (inputDir, outputDir)
    if alsoSaveChunks:
        saveChunks(inputDir, inputDir+ "/chunks")

    imageIndex = 1;
    for fileName in os.listdir(inputDir):
        path = inputDir + '/' + fileName
        f, file_extension = os.path.splitext(path)
        if not isfile(path) or file_extension != '.jpg':
            continue

        # print "- opening "+path
        chunks = getImageAndMaskDifferenceChunks(path)
        for index, c in enumerate(chunks):
            savePath = "%s/chunk-%s-%s-%s%s" % (outputDir, imageIndex, int(index % 20), int(index / 20), file_extension)
            print "chunk "+savePath
            if(chunkIsRelevant(c, relevanceThreshold)):
                c.save(savePath)
        imageIndex+=1
    print "saveDifferenceChunks - finished"

def loadChunksFromDisk(folderPath):
    print "loading chunks from disk."
    images = []
    for fileName in os.listdir(folderPath):
        filePath = folderPath + '/' + fileName

        f, file_extension = os.path.splitext(filePath)
        if not isfile(filePath) or file_extension != '.jpg':
            continue

        # print "- opening "+filePath
        c = Image.open(filePath)
        cFlattened = np.asarray(c).flatten()
        images.append(cFlattened)
        # print cFlattened
    print "loading chunks from disk. - finished"

    return images



# NETWORK STUFF

def createNetwork(inputSize, outputSize):
    if  os.path.isfile(XmlNetworkName):
        print "xml found."
        fnn = NetworkReader.readFrom(XmlNetworkName)
    else:
        print "network in/out: %s/%s " % (inputSize, outputSize)
        fnn = buildNetwork( inputSize, 10, outputSize, hiddenclass=TanhLayer, outclass=SigmoidLayer , bias=True)
    return fnn

def saveNetwork(fnn):
    NetworkWriter.writeToFile(fnn, XmlNetworkName)

def trainNetwork(fnn, data, epochs=50):
    trainer = BackpropTrainer( fnn, dataset=data, momentum=0.1, verbose=True, weightdecay=0.01)
    # for i in range(epochs):
    # trainer.trainEpochs( 50 )
    trainer.trainUntilConvergence(continueEpochs=30)
    return;

def activateOnImage(fnn, layerpath):
    chunks = getImageAndMaskDifferenceChunks(layerpath)
    fileName = os.path.basename(layerpath)

    for c in chunks:
        cFlattened = np.asarray(c).flatten()
        estimate = fnn.activate(cFlattened)
        print fileName, estimate
        if estimate > 0.7:
            print "estimated %s KO at %s " % (fileName, estimate)


    return

def generateDataset():
    trainingfullDataset = ClassificationDataSet(chunkSize, nb_classes=2, class_labels=['Ok','Non Ok'])

    flattenedOkSamples = loadChunksFromDisk(sampleDataOkPath + "/chunks_differences")
    for f in flattenedOkSamples:
        trainingfullDataset.addSample(f,[0])

    flattenedKoSamples = loadChunksFromDisk(sampleDataKoPath + "/chunks_differences")
    for f in flattenedKoSamples:
        trainingfullDataset.addSample(f,[1])

    return trainingfullDataset


#saveChunks()
# saveDifferenceChunks(sampleDataOkPath, sampleDataOkPath + "/chunks_differences", True)
# saveDifferenceChunks(sampleDataKoPath, sampleDataKoPath + "/chunks_differences", False, 5)
# exit()


trainingfullDataset = generateDataset()
fnn = createNetwork(trainingfullDataset.indim, trainingfullDataset.outdim)
trainNetwork(fnn, trainingfullDataset, 50)
saveNetwork(fnn)


# fnn = NetworkReader.readFrom(XmlNetworkName)
# print "OKtesting:"
# activateOnImage(fnn,testDataPath+"/ok1.jpg")
# activateOnImage(fnn,testDataPath+"/ok2.jpg")
# activateOnImage(fnn,testDataPath+"/ok3.jpg")
# activateOnImage(fnn,testDataPath+"/ok4.jpg")

# print "KO testing:"
# activateOnImage(fnn,testDataPath+"/ko1.jpg")
# activateOnImage(fnn,testDataPath+"/ko2.jpg")
# activateOnImage(fnn,testDataPath+"/ko3.jpg")
# activateOnImage(fnn,testDataPath+"/ko4.jpg")


# aggiungere parametri da linea di comando:
# http://www.tutorialspoint.com/python/python_command_line_arguments.htm

