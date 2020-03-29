import random
from main_gpu_win import *
from evaluate import *

_subset = ['10', '1', '2', '3', '4', '5', '6', '7','8','9']
model_name = 'GTV.pkl'

for i in range(1, 6):
    random.seed(i)
    subset = random.sample(_subset, 5)
    testset = [i for i in _subset if i not in subset]
    print("Train: ", subset)
    print("Test: ", testset)
    m = '{0}_{1}'.format(i, model_name)
    o = m + 'optim'
    main(seed=i, model_name=m, optim_name = o, subset=subset, epoch=100)
    main_eva(seed=i, model_name=m, trainset=subset, testset=testset)
