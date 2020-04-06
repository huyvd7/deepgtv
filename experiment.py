import random
from evaluate import *
#from main_gpu_win import *
from main_gpu_artificial import *

_subset = ['10', '1', '2', '3', '4', '5', '6', '7','8','9']
model_name = 'GTV.pkl'

opt = OPT(batch_size = 50, admm_iter=4, prox_iter=3, delta=.1, channels=3, eta=.3, u=25, lr=8e-6, momentum=0.9, u_max=65, u_min=55)

result = dict({'psnr_train':list(), 'ssim_train':list(),
                'psnr_test':list(), 'ssim_test':list(),
                'mse_train':list(), 'mse_test':list()})

for i in range(1, 2):
    random.seed(i)
    subset = random.sample(_subset, 5)
    testset = [i for i in _subset if i not in subset]
    print("Train: ", subset)
    print("Test: ", testset)
    m = '{0}_{1}'.format(i, model_name)
    o = m + 'optim'
    cont=None
    main(seed=i, model_name=m, cont=cont, epoch=1600, subset=subset)
    #traineva, testeva = main_eva(seed=i, model_name=m, trainset=subset, testset=testset)
    traineva, testeva = main_eva(seed=i, model_name=m, trainset=subset, testset=testset, verbose=1, image_path='..\\gauss', noise_type='gauss')
    result['psnr_train'].append(traineva['psnr2'])
    result['ssim_train'].append(traineva['ssim'])
    result['psnr_test'].append(testeva['psnr2'])
    result['ssim_test'].append(testeva['ssim'])
    result['mse_train'].append(traineva['mse'])
    result['mse_test'].append(testeva['mse'])

print("+++++++++++++++++++++++++++++++")
print("+++ FINAL EVALUATION RESULT +++")
print("+++++++++++++++++++++++++++++++")
print("PSNR Train: ", np.mean(result['psnr_train']))
print("SSIM Train: ", np.mean(result['ssim_train']))
print("PSNR Test: ", np.mean(result['psnr_test']))
print("SSIM Test: ", np.mean(result['ssim_test']))
print("MSE Train: ", np.mean(result['mse_train']))
print("MSE Test: ", np.mean(result['mse_test']))
print("+++++++++++++++++++++++++++++++")
print("+++++++++++ DETAILS +++++++++++")
#print("PSNR TRAIN: ")
#for i, v in enumerate(result['psnr_train']):
#    print("#{0}: {1:.5f} | ".format(i, v), end=' ')
#print()
#print("PSNR TEST: ")
#for i, v in enumerate(result['psnr_test']):
#    print("#{0}: {1:.5f} | ".format(i, v), end=' ')
#print()
#print("MSE TRAIN: ")
#for i, v in enumerate(result['mse_train']):
#    print("#{0}: {1:.5f} | ".format(i, v), end=' ')
#print()
#print("MSE TEST: ")
#for i, v in enumerate(result['mse_test']):
#    print("#{0}: {1:.5f} | ".format(i, v), end=' ')
#print()
#print("SSIM TRAIN: ")
#for i, v in enumerate(result['ssim_train']):
#    print("#{0}: {1:.5f} | ".format(i, v), end=' ')
#print()
#print("SSIM TEST: ")
#for i, v in enumerate(result['ssim_test']):
#    print("#{0}: {1:.5f} | ".format(i, v), end=' ')
import pandas as pd
from tabulate import tabulate
newd = dict()
for i,v in result.items():
    newd[i] = [np.mean(j) for j in v]
df = pd.DataFrame.from_dict(newd)
print(tabulate(df, headers='keys', floatfmt=".5f"))
print("+++++++++++++++++++++++++++++++")
