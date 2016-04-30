#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import SigmoidLayer
from pybrain.utilities           import percentError
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

import random

basePath = os.path.dirname(os.path.realpath(__file__))
trainingDataPath = basePath + '/ducati/training'
size = 480 * 270

def incasinanomi():
    eee = trainingDataPath + '/wrongdata/masks'
    p = []
    for image in os.listdir(eee):
        p.append(eee+'/'+image)
        print image

    random.shuffle(p)
    print p


    for index,path in enumerate(p):
        tempname = os.path.dirname(path)+ '/temp.png'
        next_position = index+1
        if(next_position == len(p)):
            next_position = 0
        os.rename(path, tempname)
        os.rename(p[next_position], path)
        os.rename(tempname, p[next_position])

    exit()

positions = ["ee", "ne", "nn", "nw", "ww", "sw", "ss", "se", "tp", "sk"]
def loadImages(path):
    trainingLayersPath = path + '/color'
    trainingMasksPath = path + '/masks'

    result = []
    for image in os.listdir(trainingLayersPath):
        if image == '.DS_Store':
            continue

        layerPath = trainingLayersPath + "/" + image
        maskPath = trainingMasksPath + "/" + image.replace('jpg','png')
        if not os.path.isfile(maskPath):
            print "Mask not found: "+maskPath
        else:
            l = Image.open(layerPath)
            m = Image.open(maskPath)

            # converto il layer in b/w
            bwl = getBlackAndWhite(l)

            layerBWImageAsArray = np.asarray(bwl)

            #prendo solo il canale alfa (?)
            maskAsArray = np.asarray(m.split()[1])

            # print "flattened: layer: %s mask: %s " % (len(layerBWImageAsArray.flatten()), len(maskAsArray.flatten()))

            lf = layerBWImageAsArray.flatten()
            mf = maskAsArray.flatten()


            # faccio la differenza tra maschera e immagine, per usare questo dato come input
            # invece delle 2 immagini separate
            diffArray = [ mf[x] -lf[x] for x in mf]


            # merge = np.concatenate([layerBWImageAsArray.flatten(), maskAsArray.flatten()])
            # result.append(merge)

            result.append(diffArray)
            # showImages(l,m.split()[1],bwl)
            # exit()

    return result


def getBlackAndWhite(img):
    gray = img.convert('L')
    bw = gray.point(lambda x: 0 if x<150 else 255)
    return bw

def showImages(img1, img2, img3=False, img4=False):

    fig = plt.figure()
    ax1 = plt.subplot2grid((8,6),(0,0), rowspan=4, colspan=3) # top-left
    ax2 = plt.subplot2grid((8,6),(4,0), rowspan=4, colspan=3) # bottom-left
    ax3 = plt.subplot2grid((8,6),(0,3), rowspan=4, colspan=3) # top-right
    ax4 = plt.subplot2grid((8,6),(4,3), rowspan=4, colspan=3) # bottom-right



    ax1.imshow(img1)
    ax3.imshow(img2)
    if img3:
        ax2.imshow(img3, cmap = matplotlib.cm.Greys_r)
    if img4:
        ax4.imshow(img4, cmap = matplotlib.cm.Greys_r)

    plt.show()

    # np.array(img1)
    # np.array(img2)

    # ax1.imshow(
    #         )
    # ax2.imshow()

def testNetwork(layerPath, maskPath):
    return

# dati per training e test
trainingfullDataset = ClassificationDataSet(size,nb_classes=2, class_labels=['Ok','Non Ok'])
ok_samples = loadImages(trainingDataPath+'/ee')
wrong_samples = loadImages(trainingDataPath+'/wrongdata')

# add training data to dataset
for s in ok_samples:
    # print "sample len: %s - expected: %s " % (len(s), size*2)
    trainingfullDataset.addSample(s, [0])

for s in wrong_samples:
    trainingfullDataset.addSample(s, [1])

tstdata, trndata = trainingfullDataset.splitWithProportion( 0.25 )


trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )


# dati per activation
# testDataset = ClassificationDataSet(2 * size,nb_classes=2, class_labels=['Ok','Non Ok'])
test_samples = loadImages(trainingDataPath+'/testdata')

# for s in test_samples:
#     testDataset.addSample(s)

# testDataset._convertToOneOfMany( )



# print "Number of training patterns: ", len(trainingfullDataset)
# print "Input and output dimensions: ", trainingfullDataset.indim, trainingfullDataset.outdim
# print "First sample (input, target, class):"
# print trainingfullDataset['input'][0], trainingfullDataset['target'][0], trainingfullDataset['class'][0]

# build and train network
if  os.path.isfile('ducati.xml'):
    print "xml found."
    fnn = NetworkReader.readFrom('ducati.xml')
else:
    fnn = buildNetwork( trndata.indim, 10, trndata.outdim, outclass=SigmoidLayer )

# Set up a trainer that basically takes the network and training dataset as input. For a list of trainers, see trainers. We are using a BackpropTrainer for this.
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

print "network and trainer built. starting training."


for i in range(10):
    print "iteration %s" % (i)
    trainer.trainEpochs( 5 )
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )

    tstresult = percentError( trainer.testOnClassData(
            dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

    print fnn.activate(test_samples[0])

# NetworkWriter.writeToFile(fnn, 'ducati.xml')


print "testing on samples. (sample lenght: %s) " % len(test_samples)
for index,s in enumerate(test_samples):
    print "sample %s: %s " % (index, fnn.activate(s))

#load test images

#activate network on image

