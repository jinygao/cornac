from cornac.utils import util_data
import scipy
import numpy
import random
from cornac.evaluation_strategies import Split
import pdb
from scipy.io import loadmat

# Loading and preparing the Amazon office data,

office= loadmat("office.mat")
mat_office = office['mat']

rawdata = util_data.Dataset(mat_office)
validData, validUsers, validItems = rawdata.index_trans()

es_splitraw = Split(data = mat_office, prop_test=0, prop_validation=0.0, good_rating=4)
es_splitclean = Split(data = validData, prop_test=0, prop_validation=0.0, good_rating=4)

#Prepare Patara and experiment dataset which has different Users but same Items
experimentUserSet = random.sample(validUsers, round(len(validUsers)*0.2))  # save 20% users as experiment set

indexS = numpy.isin(validData[:, 0], experimentUserSet)
indexB = numpy.isin(validData[:, 0], experimentUserSet, invert=True)
experimentData = validData[indexS, :]
pataraData = validData[indexB, :]

print("pataraData valid rating:", len(pataraData))
print("experimentData valid rating:", len(experimentData))

print("pataraData valid users:", len(numpy.unique(pataraData[:, 0])), "pataraData valid items:", len(numpy.unique(pataraData[:, 1])))
print("experimentData valid users:", len(numpy.unique(experimentData[:, 0])), "experimentData valid items:", len(numpy.unique(experimentData[:, 1])))

scipy.io.savemat('SourceData.mat', {'mat': pataraData})
scipy.io.savemat('TargetData.mat', {'mat': experimentData})

pataraData = loadmat("SourceData.mat")['mat']
experimentData = loadmat("TargetData.mat")['mat']

overlap = list(set(experimentData[:, 1]) & set(pataraData[:, 1]))
#
# initialU_patara = numpy.random.normal(loc=0.0, scale=1.0, size=len(numpy.unique(pataraData[:, 0]))*10).reshape(len(numpy.unique(pataraData[:, 0])),10).astype(numpy.float32)
# initialV_patara = numpy.random.normal(loc=0.0, scale=1.0, size=len(validItems)*10).reshape(len(validItems),10).astype(numpy.float32)
#
# initialU_experiment = numpy.random.normal(loc=0.0, scale=1.0, size=len(numpy.unique(experimentData[:, 0]))*10).reshape(len(numpy.unique(experimentData[:, 0])),10).astype(numpy.float32)
# initialV_experiment = numpy.random.normal(loc=0.0, scale=1.0, size=len(validItems)*10).reshape(len(validItems),10).astype(numpy.float32)
#
# scipy.io.savemat('patara_init_parameters.mat', {'U': initialU_patara, 'V':initialV_patara})
# scipy.io.savemat('experiment_init_parameters.mat', {'U': initialU_experiment, 'V':initialV_experiment})

initialU = numpy.random.normal(loc=0.0, scale=1.0, size=len(validUsers)*10).reshape(len(validUsers),10).astype(numpy.float32)
initialV = numpy.random.normal(loc=0.0, scale=1.0, size=len(validItems)*10).reshape(len(validItems),10).astype(numpy.float32)

scipy.io.savemat('init_parameters.mat', {'U': initialU, 'V':initialV})
