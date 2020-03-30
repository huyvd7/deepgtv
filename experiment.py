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
    lr=1e-4,
    momentum=0.9,
    u_max=75,
    u_min=25,
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
    main(seed=i, model_name=m, optim_name = o, subset=subset, epoch=100)
    _psnr_train, _ssim_train, _psnr_test, _ssim_test = main_eva(seed=i, model_name=m, trainset=subset, testset=testset, imgw=1080)
    result['psnr_train'].append(_psnr_train)
    result['ssim_train'].append(_ssim_train)
    result['psnr_test'].append(_psnr_test)
    result['ssim_test'].append(_ssim_test)

print("+++++++++++++++++++++++++++++++")
print("+++ FINAL EVALUATION RESULT +++")
print("+++++++++++++++++++++++++++++++")
print("PSNR Train: ", np.mean(result['psnr_train']))
print("SSIM Train: ", np.mean(result['ssim_train']))
print("PSNR Test: ", np.mean(result['psnr_test']))
print("SSIM Test: ", np.mean(result['ssim_test']))
print("+++++++++++++++++++++++++++++++")
