import sys
import pickle
import torch
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import argparse
from train_gtv import *
import logging

cuda = True if torch.cuda.is_available() else False
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

resroot = "result"


def denoise(
    inp,
    gtv,
    argref,
    normalize=False,
    stride=36,
    width=324,
    prefix="_",
    verbose=0,
    opt=None,
    approx=False,
    args=None,
    logger=None,
):
    try:
        from skimage.metrics import structural_similarity as compare_ssim
    except Exception:
        from skimage.measure import compare_ssim

    sample = cv2.imread(inp)
    if width is None:
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

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if argref:
        ref = cv2.imread(argref)
        if ref.shape[0] != width or ref.shape[1] != width:
            ref = cv2.resize(ref, (width, width))
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        ref_p = resroot + "/ref_" + argref.split("/")[-1]
        plt.imsave(ref_p, ref)
        logger.info(ref_p)
        tref = ref.copy()
        ref = ref.transpose((2, 0, 1))
        ref = torch.from_numpy(ref)
        if normalize:
            ref = _norm(ref, newmin=0, newmax=1)

    tstart = time.time()
    T1 = sample
    if argref:
        T1r = ref

    T1 = torch.nn.functional.pad(T1, (0, stride, 0, stride), mode="constant", value=0)
    shapex = T1.shape
    T2 = (
        torch.from_numpy(T1.detach().numpy().transpose(1, 2, 0))
        .unfold(0, opt.width, stride)
        .unfold(1, opt.width, stride)
    ).type(dtype)
    T2 = T2.contiguous()
    if argref:
        T1r = torch.nn.functional.pad(
            T1r, (0, stride, 0, stride), mode="constant", value=0
        )
    MAX_PATCH = args.multi
    oT2s0 = T2.shape[0]
    T2 = T2.view(-1, opt.channels, opt.width, opt.width)
    dummy = torch.zeros(T2.shape).type(dtype)
    logger.info("{0}".format(T2.shape))
    with torch.no_grad():
        for ii, i in enumerate(range(0, T2.shape[0], MAX_PATCH)):
            P = gtv.predict(
                T2[i : (i + MAX_PATCH), :, :, :].float().contiguous(),
                layers=args.layers,
            )
            dummy[i : (i + MAX_PATCH)] = P
    dummy = dummy.view(oT2s0, -1, opt.channels, opt.width, opt.width)
    dummy = dummy.cpu()
    if verbose:
        logger.info("Prediction time: {0}".format(time.time() - tstart))
    else:
        logger.info("Prediction time: {0}".format(time.time() - tstart))

    dummy = (
        patch_merge(dummy, stride=stride, shape=shapex, shapeorg=shape).detach().numpy()
    )

    ds = np.array(dummy).copy()
    d = np.minimum(np.maximum(ds, 0), 255)
    logger.info("RANGE: {0} - {1}".format(d.min(), d.max()))
    d = d.transpose(1, 2, 0) / 255
    if 0:
        opath = args.output
    else:
        filename = inp.split("/")[-1]
        opath = resroot + "/{0}_{1}".format(prefix, filename)
        opath = opath[:-3] + "png"
    d = np.minimum(np.maximum(d, 0), 1)
    plt.imsave(opath, d)
    if argref:
        mse = ((d - (tref / 255.0)) ** 2).mean() * 255
        logger.info("MSE: {:.5f}".format(mse))
        d = cv2.imread(opath)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        psnr2 = cv2.PSNR(tref, d)
        logger.info("PSNR: {:.5f}".format(psnr2))
        (score, diff) = compare_ssim(tref, d, full=True, multichannel=True)
        logger.info("SSIM: {:.5f}".format(score))
    logger.info("Saved {0}".format(opath))
    if argref:

        return (0, score, 0, psnr2, mse, d)  # psnr, ssim, denoised image
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

            R[:, ri : (ri + m), rj : (rj + m)] += P[i, j, :, :, :].cpu()
            Rc[:, ri : (ri + m), rj : (rj + m)] += 1
            rj += stride
            c += 1
        ri += stride
        rj = 0

    return (R / Rc)[:, : shapeorg[-1], : shapeorg[-1]]


def main_eva(
    seed,
    model_name,
    trainset,
    testset,
    imgw=None,
    verbose=0,
    image_path=None,
    noise_type="gauss",
    opt=None,
    args=None,
    logger=None,
):
    gtv = GTV(width=36, cuda=cuda, opt=opt)  # just initialize to load the trained model, no need to change
    PATH = model_name
    device = torch.device("cuda") if cuda else torch.device("cpu")
    gtv.load_state_dict(torch.load(PATH, map_location=device))
    width = gtv.opt.width
    opt.width = width
    opt = gtv.opt
    if not image_path:
        image_path = "..\\all\\all\\"
    if noise_type == "gauss":
        npref = "_g"
    else:
        npref = "_n"

    logger.info("EVALUATING TRAIN SET")
    # trainset = ["10", "1", "7", "8", "9"]
    traineva = {
        "psnr": list(),
        "ssim": list(),
        "ssim2": list(),
        "psnr2": list(),
        "mse": list(),
    }
    stride = args.stride
    for t in trainset:
        logger.info("image #{0}".format(t))
        inp = "{0}/noisy/{1}{2}.bmp".format(image_path, t, npref)
        logger.info(inp)
        argref = "{0}/ref/{1}_r.bmp".format(image_path, t)
        _, _ssim, _, _psnr2, _mse, _ = denoise(
            inp,
            gtv,
            argref,
            stride=stride,
            width=imgw,
            prefix=seed,
            opt=opt,
            args=args,
            logger=logger,
        )
        # traineva["psnr"].append(_psnr)
        traineva["ssim"].append(_ssim)
        # traineva["ssim2"].append(_ssim2)
        traineva["psnr2"].append(_psnr2)
        traineva["mse"].append(_mse)
        try:
            from skimage.metrics import structural_similarity as compare_ssim
        except Exception:
            from skimage.measure import compare_ssim

        img1 = cv2.imread(inp)[:, :, : opt.channels]
        img2 = cv2.imread(argref)[:, :, : opt.channels]
        (score, diff) = compare_ssim(img1, img2, full=True, multichannel=True)
        logger.info("Original {0:.2f} {1:.2f}".format(cv2.PSNR(img1, img2), score))
    logger.info("========================")
    # logger.info("MEAN PSNR: {:.2f}".format(np.mean(traineva["psnr"])))
    logger.info("MEAN SSIM: {:.3f}".format(np.mean(traineva["ssim"])))
    # logger.info("MEAN SSIM2 (patch-based SSIM): {:.2f}".format(np.mean(traineva["ssim2"])))
    logger.info(
        "MEAN PSNR2 (image-based PSNR): {:.2f}".format(np.mean(traineva["psnr2"]))
    )
    logger.info("MEAN MSE (image-based MSE): {:.2f}".format(np.mean(traineva["mse"])))
    logger.info("========================")

    logger.info("EVALUATING TEST SET")

    # testset = ["2", "3", "4", "5", "6"]
    testeva = {
        "psnr": list(),
        "ssim": list(),
        "ssim2": list(),
        "psnr2": list(),
        "mse": list(),
    }
    for t in testset:
        logger.info("image #{0}".format(t))
        inp = "{0}/noisy/{1}{2}.bmp".format(image_path, t, npref)
        logger.info(inp)
        argref = "{0}/ref/{1}_r.bmp".format(image_path, t)
        _, _ssim, _, _psnr2, _mse, _ = denoise(
            inp,
            gtv,
            argref,
            stride=stride,
            width=imgw,
            prefix=seed,
            opt=opt,
            args=args,
            logger=logger,
        )
        # testeva["psnr"].append(_psnr)
        testeva["ssim"].append(_ssim)
        # testeva["ssim2"].append(_ssim2)
        testeva["psnr2"].append(_psnr2)
        testeva["mse"].append(_mse)
        try:
            from skimage.metrics import structural_similarity as compare_ssim
        except Exception:
            from skimage.measure import compare_ssim

        img1 = cv2.imread(inp)[:, :, : opt.channels]
        img2 = cv2.imread(argref)[:, :, : opt.channels]
        (score, diff) = compare_ssim(img1, img2, full=True, multichannel=True)
        logger.info("Original {0:.2f} {1:.2f}".format(cv2.PSNR(img1, img2), score))
    logger.info("========================")
    # logger.info("MEAN PSNR: {:.2f}".format(np.mean(testeva["psnr"])))
    logger.info("MEAN SSIM: {:.3f}".format(np.mean(testeva["ssim"])))
    # logger.info("MEAN SSIM2 (patch-based SSIM): {:.2f}".format(np.mean(testeva["ssim2"])))
    logger.info(
        "MEAN PSNR2 (image-based PSNR): {:.2f}".format(np.mean(testeva["psnr2"]))
    )
    logger.info("MEAN MSE (image-based MSE): {:.2f}".format(np.mean(testeva["mse"])))
    logger.info("========================")
    return traineva, testeva


if __name__ == "__main__":
    # global opt
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w",
        "--width",
        help="Resize image to a square image with given width",
        type=int,
    )
    parser.add_argument("-m", "--model")
    parser.add_argument("--stride", default=18, type=int)
    parser.add_argument(
        "--multi", default=30, type=int, help="# of patches evaluation in parallel"
    )
    parser.add_argument("--opt", default="opt")
    parser.add_argument("-p", "--image_path")
    parser.add_argument("--layers", default=1, type=int)
    args = parser.parse_args()
    opt = pickle.load(open(args.opt, "rb"))
    supporting_matrix(opt)
    if args.model:
        model_name = args.model
    else:
        model_name = "GTV_20.pkl"
    if args.image_path:
        image_path = args.image_path
    else:
        image_path = "gauss"
    logging.basicConfig(
        filename="log/test_gtv_{0}.log".format(time.strftime("%Y-%m-%d-%H%M")),
        filemode="a",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.NOTSET,
    )

    logger = logging.getLogger("root")
    logger.addHandler(logging.StreamHandler(sys.stdout))

    opt.logger = logger
    logger.info("GTV evaluation")
    logger.info(" ".join(sys.argv))
    _, _ = main_eva(
        seed="gauss",
        model_name=model_name,
        trainset=["1", "3", "5", "7", "9"],
        testset=["10", "2", "4", "6", "8"],
        imgw=args.width,
        verbose=1,
        image_path=image_path,
        noise_type="gauss",
        opt=opt,
        args=args,
        logger=logger,
    )

