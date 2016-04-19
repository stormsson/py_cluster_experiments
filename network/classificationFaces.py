#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from sklearn import datasets

olivetti = datasets.fetch_olivetti_faces()
X, y = olivetti.data, olivetti.target

print X.shape
# load all the python modules
from numpy import ravel
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

# then load the dataset into the neural network. The key here is we flatten the 64×64 data to one dimensional
# 4096 and then we feed the data our NN classification dataset:
ds = ClassificationDataSet(4096, 1 , nb_classes=40)
for k in xrange(len(X)):
 ds.addSample(ravel(X[k]),y[k])

# Next we split the data into 75% training and 25% test data.
tstdata, trndata = ds.splitWithProportion( 0.25 )

# and this code converts 1 output to 40 binary outputs
# see http://pybrain.org/docs/tutorial/datasets.html
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )


# Now that all the data is loaded into the neural network, we build the network and backpropagation trainer.
fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)


# Sometimes when you train for hours, you might want to save the data.
# Here is how to do it using the built in pybrain function.  Replace the above fnn variable with this.
# Here we check if the file exists we resume from where the training stopped by loading the file.

if  os.path.isfile('oliv.xml'):
 fnn = NetworkReader.readFrom('oliv.xml')
else:
 fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )


# We train our network for 50 epochs and compute the percentage error on test data.
# I am not gonna show the error on training data because it is usually about less than 2% and this not important.
# Sometimes the network overfit and memorize the data. The key here is test data set.

trainer.trainEpochs (150)
print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
           dataset=tstdata )
           , tstdata['class'] )



# save training log
NetworkWriter.writeToFile(fnn, 'oliv.xml')