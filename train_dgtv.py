import pickle
import logging
import argparse
import torch
import numpy as np
import os
import time
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image as save_image
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from dgtv.dgtv import *
import sys


def main(seed, model_name, cont=None, optim_name=None, subset=None, epoch=100):
    cuda = True if torch.cuda.is_available() else False
    torch.autograd.set_detect_anomaly(True)
    opt.logger.info("CUDA: {0}".format(cuda))
    if cuda:
        dtype = torch.cuda.FloatTensor
        opt.logger.info(torch.cuda.get_device_name(0))
    else:
        dtype = torch.FloatTensor

    DST = "./"
    DST = ""
    PATH = os.path.join(DST, model_name)
    SAVEPATH = PATH.split(".")[-1]
    SAVEDIR = "".join(PATH.split(".")[:-1]) + "_"
    batch_size = opt.batch_size

#    if not subset:
#        _subset = ["10", "1", "7", "8", "9"]
        # _subset = ["1", "3", "5", "7", "9"]
#        opt.logger.info("Train: {0}".format(_subset))
#        subset = [i + "_" for i in _subset]
    if subset:
        subset = [i + "_" for i in subset]

    # dataset = RENOIR_Dataset(
    #     img_dir=os.path.join(opt.train),
    #     transform=transforms.Compose([standardize(normalize=False), ToTensor()]),
    #     subset=None,
    # )
    # opt.logger.info("Splitting patches...")
    # patch_splitting(
    #     dataset=dataset, output_dst="tmp", patch_size=args.width, stride=args.width / 2
    # )

    dataset = RENOIR_Dataset(
        # img_dir=os.path.join("tmp", "patches"),
        img_dir=os.path.join(opt.train),
        transform=transforms.Compose([standardize(normalize=False), ToTensor()]),
        subset=subset,
    )
    opt.logger.info(dataset.nimg_name[0])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

    supporting_matrix(opt)
    total_epoch = epoch
    opt.logger.info("Dataset: {0}".format(len(dataset)))
    gtv = DeepGTV(
        width=args.width,
        prox_iter=1,
        u_max=10,
        u_min=0.5,
        cuda=cuda,
        opt=opt,
    )
    if args.stack:
        gtv.load(p1=args.stack, p2=args.stack)
        opt.logger.info("Stacked from {0}".format(args.stack))
    else:
        opt.logger.info("Train DGTV from scratch")

    if cont:
        gtv.load_state_dict(torch.load(cont))
        opt.logger.info("LOAD PREVIOUS DGTV: {0}".format(cont))
    if cuda:
        gtv.gtv1.cuda()
        gtv.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(gtv.parameters(), lr=opt.lr, momentum=opt.momentum)
    if cont:
        try:
            optimizer.load_state_dict(torch.load(cont + "optim"))
            opt.logger.info("LOAD PREVIOUS OPTIMIZER: {0}".format(cont + "optim"))
        except Exception:
            opt.logger.info("Using new optimizer")

    losshist = list()
    tstart = time.time()
    tprev = tstart
    opt._print()
    pickle.dump(opt, open("dopt", "wb"))
    ld = 1
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):  # start index at 0
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["nimg"][:, : opt.channels, :, :].float().type(dtype)
            labels = data["rimg"][:, : opt.channels, :, :].float().type(dtype)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = gtv(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gtv.parameters(), 5e1)

            optimizer.step()
            running_loss += loss.item()

            if epoch == 0 and (i + 1) % 80 == 0 and args.first: # for the first epoch, print every 80-th patch if 'first' flag is set
                g = gtv.gtv1
                with torch.no_grad():
                    P1, P2  = gtv(inputs, debug=True)
                    opt.logger.info(
                        "\tLOSS: {0:.8f} {1:.8f} ".format(
                            (P1 - labels).square().mean().item(),
                            (P2 - labels).square().mean().item()
                        )
                    )
                    P1 = g(inputs, debug=1)
                opt.logger.info(
                        "\tCNNF grads: {0:.5f}".format(
                            g.cnnf.layer[0].weight.grad.median().item()
                        )
                    )
                with torch.no_grad():
                    P2 = g(P1, debug=1)

        tnow = time.time()
        opt.logger.info(
            "[{0}] \x1b[31mLOSS\x1b[0m: {1:.8f}, time elapsed: {2:.1f} secs, epoch time: {3:.1f} secs".format(
                epoch + 1, running_loss / (ld * (i + 1)), tnow - tstart, tnow - tprev
            )
        )
        tprev = tnow

        if ((epoch + 1) % 1 == 0) or (epoch + 1) == total_epoch:
            g = gtv.gtv1
            with torch.no_grad():
                P1, P2 = gtv(inputs, debug=True)
                opt.logger.info(
                    "\tLOSS: {0:.8f} {1:.8f} ".format(
                        (P1 - labels).square().mean().item(),
                        (P2 - labels).square().mean().item()
                    )
                )

                _ = g(inputs, debug=1)
            opt.logger.info(
                    "\tCNNF grads: {0:.5f}".format(
                        g.cnnf.layer[0].weight.grad.median().item()
                    )
                )
            with torch.no_grad():
                P2 = g(P1, debug=1)

            opt.logger.info("\tsave @ epoch {0}".format(epoch + 1))
            torch.save(gtv, SAVEDIR + str(epoch) + "." + SAVEPATH)
            torch.save(
                optimizer.state_dict(), SAVEDIR + str(epoch) + "." + SAVEPATH + "optim"
            )

        # scheduler.step()
        losshist.append(running_loss / (ld * (i + 1)))
        torch.save(gtv, SAVEDIR + str(epoch) + "." + SAVEPATH)
    torch.save(optimizer.state_dict(), SAVEDIR + str(epoch) + "." + SAVEPATH + "optim")

    opt.logger.info("Total running time: {0:.3f}".format(time.time() - tstart))

opt = OPT(
    batch_size=32,
    channels=3,
    lr=1e-4,
    momentum=0.9,
    u_max=1000,
    u_min=0.0001,
    cuda=True if torch.cuda.is_available() else False
)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default="DGTV.pkl")
    parser.add_argument("-c", "--cont", default=None)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--umax", default=1000, type=float)
    parser.add_argument("--umin", default=0.001, type=float)
    parser.add_argument("--seed", default=0, type=float)
    parser.add_argument("--train", default="gauss_batch")
    parser.add_argument("--stack", default=None)
    parser.add_argument("--width", default=36, type=int)
    parser.add_argument("--legacy", default=True, type=bool, help='original architecture')
    parser.add_argument("--first", default=False, type=bool, help='print logs for the first epoch')

    args = parser.parse_args()
    model_name = args.model
    opt.batch_size = int(args.batch)
    opt.lr = float(args.lr)
    opt.u_min = args.umin
    opt.u_max = args.umax
    opt.legacy = args.legacy
    opt.ver = True
    opt.train = args.train
    opt.width = args.width
    torch.manual_seed(args.seed)
    logging.basicConfig(
        filename="log/train_dgtv_{0}.log".format(time.strftime("%Y-%m-%d-%H%M")),
        filemode="a",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.NOTSET,
    )

    logger = logging.getLogger("root")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    opt.logger = logger
    logger.info("Train DGTV")
    logger.info(" ".join(sys.argv))
    pickle.dump(opt, open("opt", "wb"))
    main(
        seed=args.seed,
        model_name=args.model,
        cont=args.cont,
        epoch=int(args.epoch),
        subset=["1", "3", "5", "7", "9"],
    )

