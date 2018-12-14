from cornac.models import Pmf
from cornac.experiment import Experiment
from cornac.evaluation_strategies import Split
from cornac import metrics
from cornac.utils import util_data
import scipy
import numpy
import random

from scipy.io import loadmat

office = loadmat("pataraTest/TargetData.mat")
data = numpy.r_[office['mat'],numpy.array([[3702,6522,1]])]

es_split = Split(data = data, prop_test=0.2, prop_validation=0.0, good_rating=4)

# #load  inital U and V
# inital_parameters= loadmat("pataraTest/init_parameters.mat")
# init_U = inital_parameters['U']
# init_V = inital_parameters['V']


#load pre-trained U and V
inital_parameters= loadmat("pataraTest/pretrainedbysource(minRMSE).mat")
init_U = inital_parameters['U'].toarray()
init_V = inital_parameters['V'].toarray()


# Instantiate evaluation metrics.
rec = metrics.Recall(m=20)
pre = metrics.Precision(m=20)
mae = metrics.Mae()
rmse = metrics.Rmse()


# train the models
bestRMSE= 1.5
for lamda in numpy.arange(0.01, 0.05, 0.01):

    this_U = numpy.copy(init_U)
    this_V = numpy.copy(init_V)
    print(this_U[0,:])
    rec_pmf = Pmf(k=10, max_iter=2, learning_rate=0.001, lamda=lamda, init_params={'U': this_U, 'V': this_V})

    # Instantiate and then run an experiment.
    res_pmf = Experiment(es_split, [rec_pmf], metrics=[mae, rmse, pre, rec])

    res_pmf.run_()

    # print(res_pmf.res_avg)

    if res_pmf.res_avg.get_values()[0][1]<bestRMSE :
        bestRMSE = res_pmf.res_avg.get_values()[0][1]
        # print("best RMSE ", bestRMSE)
        # scipy.io.savemat('sourcetotarget(minRMSE).mat', {'U': rec_pmf.U, 'V': rec_pmf.V, 'lamda': lamda,'RMSE':bestRMSE})


