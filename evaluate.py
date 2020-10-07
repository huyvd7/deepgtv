import scipy.sparse as ss
import sys
import pickle
import torch
import numpy as np
import os
import time
import cv2
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from main_gpu_artificial import *
import logging

cuda = True if torch.cuda.is_available() else False
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def denoise(inp, gtv, argref, normalize=False, stride=36, width=324, prefix='_', verbose=0, Tmod=9, opt=None, approx=False, args=None, logger=None):
    try:
        from skimage.metrics import structural_similarity as compare_ssim
    except Exception:
        from skimage.measure import compare_ssim

    sample = cv2.imread(inp)
    #print(inp)
    if width==None:
        width = sample.shape[0]
    else:
        sample = cv2.resize(sample, (width, width))
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample = sample.transpose((2, 0, 1))
    shape = sample.shape

    if normalize:
        sample = _norm(sample, newmin=0, newmax=1)
    sample = torch.from_numpy(sample)

    cuda = True if torch.cuda.is_available() else False

    device = torch.device("cuda") if cuda else torch.device("cpu")

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    psnrs = list()
    score2 = list()
    if argref:
        ref = cv2.imread(argref)
        if ref.shape[0] != width or ref.shape[1] != width:
            ref = cv2.resize(ref, (width, width))
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        tref = ref.copy()
        ref = ref.transpose((2, 0, 1))
        ref = torch.from_numpy(ref)
        if normalize:
            ref = _norm(ref, newmin=0, newmax=1)

    tstart = time.time()
    T1 = sample
    if argref:
        T1r = ref

    m = T1.shape[-1]
    T1 = torch.nn.functional.pad(T1, (0, stride, 0, stride), mode="constant", value=0)
    shapex = T1.shape
    T2 = (
        torch.from_numpy(T1.detach().numpy().transpose(1, 2, 0))
        .unfold(0, opt.width, stride)
        .unfold(1, opt.width, stride)
    ).type(dtype)

    if argref:
        T1r = torch.nn.functional.pad(
            T1r, (0, stride, 0, stride), mode="constant", value=0
        )
        T2r = (
            torch.from_numpy(T1r.detach().numpy().transpose(1, 2, 0))
            .unfold(0, opt.width, stride)
            .unfold(1, opt.width, stride)
        )

    s2 = int(T2.shape[-1])
    MAX_PATCH = args.multi
    oT2s0=T2.shape[0]
    T2  = T2.view(-1,opt.channels,opt.width,opt.width)
    dummy = torch.zeros(T2.shape).type(dtype)
    logger.info("{0}".format(T2.shape))
    with torch.no_grad():
        #for ii, i in enumerate(range(T2.shape[1])):

        #    for jj in range(0, T2.shape[1] , MAX_PATCH):
        #        if approx:
        #            P = gtv.forward_approx(T2[i, jj:(jj+MAX_PATCH) : opt.channels, :, :].float().contiguous())
        #        else:
        #            P = gtv.predict(T2[i, jj:(jj+MAX_PATCH), : opt.channels, :, :].float().contiguous())

        #        if verbose>0:
        #            print("\r{0}, {1}/{2}".format(P.shape, ii + 1, P.shape[0]), end=" ")
        #        dummy[i, jj:(jj+MAX_PATCH)] = P
        #        del P
        for ii, i in enumerate(range(0, T2.shape[0], MAX_PATCH)):
            P = gtv.predict(T2[i:(i+MAX_PATCH),:,:,:].float().contiguous())
            dummy[i:(i+MAX_PATCH)]=P
    dummy=dummy.view(oT2s0, -1, opt.channels,opt.width,opt.width)
    dummy=dummy.cpu()
    if verbose:
        logger.info("\nPrediction time: {0}".format( time.time() - tstart))
    else:
        logger.info("\nPrediction time: {0}".format( time.time() - tstart))

    dummy = (
        patch_merge(dummy, stride=stride, shape=shapex, shapeorg=shape).detach().numpy()
    )

    ds = np.array(dummy).copy()
    new_d = list()
    d = np.minimum(np.maximum(ds, 0), 255)
    logger.info("RANGE: {0} - {1}".format(d.min(), d.max()))
    d = d.transpose(1, 2, 0)/255
    if 0:
        opath = args.output
    else:
        filename = inp.split("/")[-1]
        opath = "./{0}_{1}".format(prefix, filename)
        opath = opath[:-3] + "png"
    d = np.minimum(np.maximum(d, 0), 1)
    plt.imsave(opath, d)
    if argref:
        mse = ((d-(tref/255.0))**2).mean()*255
        logger.info("MSE: {:.5f}".format(mse))
        d = cv2.imread(opath)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        psnr2 = cv2.PSNR(tref,d)
        logger.info("PSNR: {:.5f}".format(psnr2))
        (score, diff) = compare_ssim(tref, d, full=True, multichannel=True)
        logger.info("SSIM: {:.5f}".format(score))
    logger.info("Saved {0}".format( opath))
    if argref:

        return (
            0, score, 0, psnr2 , mse, d
        )  # psnr, ssim, denoised image
    return d


def patch_merge(P, stride=36, shape=None, shapeorg=None):
    S1, S2 = P.shape[0], P.shape[1]
    m = P.shape[-1]

    R = torch.zeros(shape)
    Rc = torch.zeros(shape)

    ri, rj = 0, 0
    c = 1

    for i in range(S1):
        for j in range(S2):

            R[:, ri:(ri + m), rj:(rj + m)] += P[i, j, :, :, :].cpu()
            Rc[:, ri:(ri + m), rj:(rj + m)] += 1
            rj += stride
            c += 1
        ri += stride
        rj = 0

    return (R / Rc)[:, : shapeorg[-1], : shapeorg[-1]]

def main_eva(seed, model_name, trainset, testset, imgw=None, verbose=0, image_path=None, noise_type='gauss', Tmod=None, opt=None, args=None, logger=None):
    # INITIALIZE
    #global opt
    opt.width=args.train_width
    supporting_matrix(opt)
    opt._print()
    width = args.train_width
    gtv = GTV(
        width=width,
        prox_iter=1,
        u_max=10,
        u_min=0.5,
        lambda_min=0.5,
        lambda_max=1e9,
        cuda=cuda,
        opt=opt
        
    )
    PATH = model_name
    device = torch.device("cuda") if cuda else torch.device("cpu")
    gtv.load_state_dict(torch.load(PATH, map_location=device))
    width = gtv.opt.width
    opt.width=width
    if not image_path:
        image_path = "..\\all\\all\\"
    if noise_type=='gauss':
        npref = '_g'
    else:
        npref ='_n'

    logger.info("EVALUATING TRAIN SET")
    #trainset = ["10", "1", "7", "8", "9"]
    traineva = {'psnr':list(), 'ssim':list(), 'ssim2':list(), 'psnr2':list(), 'mse':list()}
    stride=args.stride
    for t in trainset:
        logger.info("image #{0}".format( t))
        inp = "{0}/noisy/{1}{2}.bmp".format(image_path, t, npref)
        logger.info(inp)
        argref = "{0}/ref/{1}_r.bmp".format(image_path, t)
        _, _ssim, _, _psnr2, _mse, _ = denoise(inp, gtv, argref, stride=stride, width=imgw, prefix=seed, opt=opt, args=args, logger=logger)
        #traineva["psnr"].append(_psnr)
        traineva["ssim"].append(_ssim)
        #traineva["ssim2"].append(_ssim2)
        traineva['psnr2'].append(_psnr2)
        traineva['mse'].append(_mse)
        try:
            from skimage.metrics import structural_similarity as compare_ssim
        except Exception:
            from skimage.measure import compare_ssim
    
        img1 = cv2.imread(inp)[:, :, : opt.channels]
        img2 = cv2.imread(argref)[:, :, : opt.channels]
        (score, diff) = compare_ssim(img1, img2, full=True, multichannel=True)
        logger.info("Original {0:.2f} {1:.2f}".format( cv2.PSNR(img1, img2), score))
    logger.info("========================")
    #logger.info("MEAN PSNR: {:.2f}".format(np.mean(traineva["psnr"])))
    logger.info("MEAN SSIM: {:.2f}".format(np.mean(traineva["ssim"])))
    #logger.info("MEAN SSIM2 (patch-based SSIM): {:.2f}".format(np.mean(traineva["ssim2"])))
    logger.info("MEAN PSNR2 (image-based PSNR): {:.2f}".format(np.mean(traineva['psnr2'])))
    logger.info("MEAN MSE (image-based MSE): {:.2f}".format(np.mean(traineva['mse'])))
    logger.info("========================")
    
    logger.info("EVALUATING TEST SET")


    #testset = ["2", "3", "4", "5", "6"]
    testeva = {'psnr':list(), 'ssim':list(), 'ssim2':list(), 'psnr2':list(), 'mse':list()}
    for t in testset:
        logger.info("image #{0}".format( t))
        inp = "{0}/noisy/{1}{2}.bmp".format(image_path, t, npref)
        logger.info(inp)
        argref = "{0}/ref/{1}_r.bmp".format(image_path, t)
        _, _ssim, _, _psnr2, _mse, _ = denoise(inp, gtv, argref, stride=stride, width=imgw, prefix=seed, opt=opt, args=args, logger=logger)
        #testeva["psnr"].append(_psnr)
        testeva["ssim"].append(_ssim)
        #testeva["ssim2"].append(_ssim2)
        testeva['psnr2'].append(_psnr2)
        testeva['mse'].append(_mse)
        try:
            from skimage.metrics import structural_similarity as compare_ssim
        except Exception:
            from skimage.measure import compare_ssim
    
        img1 = cv2.imread(inp)[:, :, : opt.channels]
        img2 = cv2.imread(argref)[:, :, : opt.channels]
        (score, diff) = compare_ssim(img1, img2, full=True, multichannel=True)
        logger.info("Original {0:.2f} {1:.2f}".format( cv2.PSNR(img1, img2), score))
    logger.info("========================")
    #logger.info("MEAN PSNR: {:.2f}".format(np.mean(testeva["psnr"])))
    logger.info("MEAN SSIM: {:.2f}".format(np.mean(testeva["ssim"])))
    #logger.info("MEAN SSIM2 (patch-based SSIM): {:.2f}".format(np.mean(testeva["ssim2"])))
    logger.info("MEAN PSNR2 (image-based PSNR): {:.2f}".format(np.mean(testeva['psnr2'])))
    logger.info("MEAN MSE (image-based MSE): {:.2f}".format(np.mean(testeva['mse'])))
    logger.info("========================")
    return traineva, testeva
if __name__=="__main__":
    #global opt
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-w", "--width", help="Resize image to a square image with given width"
    )
    parser.add_argument(
        "-m", "--model"
    )
    parser.add_argument(
        "--stride", default=18, type=int
    )
    parser.add_argument(
        "--multi", default=30, type=int, help='# of patches evaluation in parallel'
    )
    parser.add_argument(
        "--train_width", default=36, type=int, help='patch size that GTV was trained'
    )
    parser.add_argument(
        "--opt", default='opt'
    )
    parser.add_argument(
        "-p", "--image_path"
    )
    parser.add_argument(
        "--delta", default=0.9
    )
    args = parser.parse_args()
    opt = pickle.load(open(args.opt, "rb"))
    if args.width:
        imgw = int(args.width)
    else:
        imgw = None
    supporting_matrix(opt)
    if args.model:
        model_name = args.model
    else:
        model_name = 'GTV_20.pkl'
    if args.image_path:
        image_path = args.image_path
    else:
        image_path = 'gauss'
    opt.delta = float(args.delta)
    logging.basicConfig(filename='evaluate_{0}.log'.format(time.strftime("%Y-%m-%d-%H%M")),
                            filemode='a',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.NOTSET)

    logger = logging.getLogger('root')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    opt.logger=logger
    logger.info("GTV evaluation")
    _, _ = main_eva(seed='gauss', model_name=model_name, trainset=['1', '3', '5', '7', '9'], testset=['10', '2', '4', '6', '8'],imgw=imgw, verbose=1, image_path=image_path, noise_type='gauss', Tmod=0, opt=opt, args=args, logger=logger)
