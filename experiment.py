import random
from evaluate import *
#from main_gpu_win import *
from main_gpu_artificial import *

_subset = ['10', '1', '2', '3', '4', '5', '6', '7','8','9']
model_name = 'GTV.pkl'
opt = OPT(
    batch_size=50,
    admm_iter=4,
    prox_iter=3,
    delta=0.1,
    channels=3,
    eta=0.3,
    u=25,
    lr=1e-5,
    momentum=0.9,
    u_max=75,
    u_min=50,
)
result = dict({'psnr_train':list(), 'ssim_train':list(),
                'psnr_test':list(), 'ssim_test':list()})

for i in range(1, 6):
    random.seed(i)
    subset = random.sample(_subset, 5)
    testset = [i for i in _subset if i not in subset]
    print("Train: ", subset)
    print("Test: ", testset)
    m = '{0}_{1}'.format(i, model_name)
    o = m + 'optim'
    cont=None
    #main(seed=i, model_name=m, optim_name = o, subset=subset, epoch=100)
    main(seed=i, model_name=m, cont=cont, epoch=1, subset=subset)
    traineva, testeva = main_eva(seed=i, model_name=m, trainset=subset, testset=testset)
    result['psnr_train'].append(traineva['psnr2'])
    result['ssim_train'].append(traineva['ssim'])
    result['psnr_test'].append(testeva['psnr2'])
    result['ssim_test'].append(testeva['ssim'])

print("+++++++++++++++++++++++++++++++")
print("+++ FINAL EVALUATION RESULT +++")
print("+++++++++++++++++++++++++++++++")
print("PSNR Train: ", np.mean(result['psnr_train']))
print("SSIM Train: ", np.mean(result['ssim_train']))
print("PSNR Test: ", np.mean(result['psnr_test']))
print("SSIM Test: ", np.mean(result['ssim_test']))
print("+++++++++++++++++++++++++++++++")
