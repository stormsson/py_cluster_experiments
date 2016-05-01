#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
from PIL import ImageFilter
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


def getImageAndMaskDifferenceChunks(imgPath):
    maskPath = imgPath.replace('.jpg','.png')

    if not isfile(imgPath):
        raise IOError("Image %s not found." % (imgPath))

    if not isfile(maskPath):
        raise IOError("Mask %s not found." % (maskPath))

    srcImage = Image.open(imgPath)
    maskImage = Image.open(maskPath)

    srcImage = ImageUtils.preprocessImage(srcImage,240)
    maskImage = ImageUtils.preprocessMask(maskImage, 210)

    imageChunks = ImageUtils.splitImageInChunks(srcImage, chunkWidth, chunkHeight)
    maskChunks = ImageUtils.splitImageInChunks(maskImage, chunkWidth, chunkHeight)

    # imageChunks = ImageUtils.splitImageInChunks(ImageUtils.getBlackAndWhite(srcImage), chunkWidth, chunkHeight)
    # maskChunks = ImageUtils.splitImageInChunks(ImageUtils.getPNGBWMask(maskImage, 210), chunkWidth, chunkHeight)

    return ImageUtils.getDifferenceChunks(imageChunks, maskChunks)

# 253.5
# discard after threshold if mean > threshold
# discard before threshold if mean < threshold

def chunkIsRelevant(c, threshold=250):
    stats = ImageStat.Stat(c)
    # la mediana mi indica qual'e' il valore (colore) piu frequente
    # la media del colore mi indica quanto è bianca/nera (perhe tratto solo pixel bianchi o neri)
    print " mean %s, median %s" % ( stats.mean, stats.median)

    if stats.mean[0] > threshold:
    # if stats.mean[0] < threshold:
        print "mean at %s, ignoring chunk" % (stats.mean)
        return False

    return True

def saveChunks(inputDir, outputDir, relevanceThreshold=255):

    # current chain in test: jpg => bw > EDGE_ENHANCE > invert

    print "saveChunks from %s to %s " % (inputDir, outputDir)
    imageIndex = 1
    for fileName in os.listdir(inputDir):
        path = inputDir + '/' + fileName
        f, file_extension = os.path.splitext(path)

        if file_extension == '.png':
            maskImage = ImageUtils.preprocessMask(Image.open(path),210)
            chunks = ImageUtils.splitImageInChunks(maskImage, chunkWidth, chunkHeight)
            for index, c in enumerate(chunks):
                if(chunkIsRelevant(c)):
                # if(chunkIsRelevant(c)) and imageIndex==1 and (index%20 ==7):
                    c.save("%s/chunk-%s-%s-%s%s" % (outputDir, imageIndex, int(index % 20), int(index / 20), file_extension))
            imageIndex+=1
            continue

        if not isfile(path) or file_extension != '.jpg':
            continue


        print "- opening "+path
        # chunks = getImageAndMaskDifferenceChunks(path)

        srcImage = ImageUtils.preprocessImage(Image.open(path), 240)

        ## Image processing test
        # srcImage = srcImage.filter(ImageFilter.EDGE_ENHANCE)
        # srcImage = ImageUtils.getBlackAndWhite(srcImage, 240)
        # srcImage = ImageOps.invert(srcImage)
        ##
        if file_extension == '.jpg':
            # srcImage = ImageUtils.getBlackAndWhite(srcImage)
            chunks = ImageUtils.splitImageInChunks(srcImage, chunkWidth, chunkHeight)

        # else:
        #     chunks = ImageUtils.splitImageInChunks(ImageUtils.getPNGBWMask(srcImage), chunkWidth, chunkHeight)


        for index, c in enumerate(chunks):
            savePath = "%s/chunk-%s-%s-%s%s" % (outputDir, imageIndex, int(index % 20), int(index / 20), file_extension)
            if(chunkIsRelevant(c, relevanceThreshold)):
            # if(chunkIsRelevant(c, relevanceThreshold)) and imageIndex==1 and (index%20 ==7):

                c.save(savePath, quality=95)

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
            # if(chunkIsRelevant(c, relevanceThreshold)) and imageIndex==1 and (index%20 ==7):
                c.save(savePath, quality=95)
        imageIndex+=1
    print "saveDifferenceChunks - finished"

def loadChunksFromDisk(folderPath):
    print "loading chunks from "+folderPath
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

def createNetwork(inputSize, outputSize, fileName):
    if  os.path.isfile(fileName):
        print "xml found."
        fnn = NetworkReader.readFrom(fileName)
    else:
        print "network in/out: %s/%s " % (inputSize, outputSize)
        fnn = buildNetwork( inputSize, 10, outputSize, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    return fnn

def saveNetwork(fnn, fileName):
    NetworkWriter.writeToFile(fnn, fileName)

def trainNetwork(fnn, data, epochs=50):
    trainer = BackpropTrainer( fnn, dataset=data, momentum=0.1, verbose=True, weightdecay=0.01)
    for i in range(epochs):
        trainer.trainEpochs( 1 )

    # trainer.trainUntilConvergence(continueEpochs=10)
    return;

def activateOnImage(fnn, layerpath, saveWrongChunks=False, breakOnError=True):
    try:
        chunks = getImageAndMaskDifferenceChunks(layerpath)
        pass
    except IOError as e:
        print e.strerror
        return

    fileName = os.path.basename(layerpath)

    index=1
    for c in chunks:
        cFlattened = np.asarray(c).flatten()
        estimate = fnn.activate(cFlattened)
        if estimate[0] < 0.7:
            print "estimated %s KO at %s " % (fileName, estimate)
            if saveWrongChunks:
                c.save("chunk-%s.jpg" %(index))

            if breakOnError:
                break
            index+=1
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


# generate initial images data
#saveChunks()
# saveDifferenceChunks(sampleDataOkPath, sampleDataOkPath + "/chunks_differences", True)
# saveDifferenceChunks(sampleDataKoPath, sampleDataKoPath + "/chunks_differences", False, 240)
# exit()

# generate dataset
# trainingfullDataset = generateDataset()
# tstdata, trndata = trainingfullDataset.splitWithProportion( 0.25 )
# trndata._convertToOneOfMany( )
# tstdata._convertToOneOfMany( )


XmlNetworkName = "subdivide32HiddenSigmoidOutSoftmax-20.xml"


# create and train network
# fnn = createNetwork(trainingfullDataset.indim, trainingfullDataset.outdim, XmlNetworkName)
# trainNetwork(fnn, trainingfullDataset, 100)
# fnn = createNetwork(trndata.indim, trndata.outdim)

# temporary test
# fnn = buildNetwork( trndata.indim, 32, trndata.outdim,  hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)

# trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
# for i in range(20):
#     trainer.trainEpochs( 1 )

# trainer.trainEpochs( 100 )
#     print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
#                dataset=tstdata )
#                , tstdata['class'] )
# saveNetwork(fnn, XmlNetworkName)


def fullTest():
    fnn = NetworkReader.readFrom(XmlNetworkName)
    for fileName in os.listdir(imagePath+"/test/sw"):
        path = imagePath+"/test/sw/"+fileName
        f, file_extension = os.path.splitext(path)
        if not isfile(path) or file_extension != '.jpg':
            continue
        activateOnImage(fnn, path, False)

fullTest()
# real test

# fnn = NetworkReader.readFrom(XmlNetworkName)
# print "OKtesting:"
# activateOnImage(fnn,testDataPath+"/ok1.jpg",False)
# activateOnImage(fnn,testDataPath+"/ok2.jpg",False)
# activateOnImage(fnn,testDataPath+"/ok3.jpg",False)
# activateOnImage(fnn,testDataPath+"/ok4.jpg",False)

# # print "KO testing:"
# activateOnImage(fnn,testDataPath+"/ko1.jpg",False)
# activateOnImage(fnn,testDataPath+"/ko2.jpg",False)
# activateOnImage(fnn,testDataPath+"/ko3.jpg",False)
# activateOnImage(fnn,testDataPath+"/ko4.jpg",False)


# aggiungere parametri da linea di comando:
# http://www.tutorialspoint.com/python/python_command_line_arguments.htm

