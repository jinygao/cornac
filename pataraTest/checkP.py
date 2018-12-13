from scipy.io import loadmat
from cornac.models import Pmf
from cornac.experiment import Experiment
from cornac.evaluation_strategies import Split
from cornac import metrics
from scipy import sparse
from cornac.utils import util_data
import scipy
import numpy
import random


rec = metrics.Recall(m=20)
pre = metrics.Precision(m=20)
mae = metrics.Mae()
rmse = metrics.Rmse()

office1= loadmat("TargetData.mat")
sourcedata = office1['mat']

es_split1 = Split(data = sourcedata, prop_test=0.2, prop_validation=0.0, good_rating=4)

#load U and V
parameters= loadmat("pretrainedbytarget(minRMSE).mat")
U = parameters['U'].toarray()
V = parameters['V'].toarray()
lamda = parameters['lamda']

rec_pmf = Pmf(k=10, max_iter=0, learning_rate=0.001, lamda=lamda, init_params={'U': U, 'V': V})
res_pmf1 = Experiment(es_split1, [rec_pmf], metrics=[mae, rmse, pre, rec])
res_pmf1.run_()

office2= loadmat("TargetData.mat")
targetdata = numpy.r_[office2['mat'],numpy.array([[3702,6522,1]])]

es_split2 = Split(data = targetdata, prop_test=0.2, prop_validation=0.0, good_rating=4)

# parameters= loadmat("Parametersbytarget(minRMSE).mat")
# U = parameters['U'].toarray()
# V = parameters['V'].toarray()
# lamda = parameters['lamda']
#
# rec_pmf = Pmf(k=10, max_iter=0, learning_rate=0.001, lamda=lamda, init_params={'U': U, 'V': V})
# res_pmf2 = Experiment(es_split2, [rec_pmf], metrics=[mae, rmse, pre, rec])
# res_pmf2.run_()
#
# print("source data trained parameter, result on source data \n", res_pmf1.res_avg)
# print("target data trained parameter, result on target data \n", res_pmf2.res_avg)