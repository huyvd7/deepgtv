import scipy.sparse as ss
import pickle
import logging
import argparse
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
from proxgtv.proxgtv import * 
import sys

def main(seed, model_name, cont=None, optim_name=None, subset=None, epoch=100):
    debug = 0

    xd = None
    cuda = True if torch.cuda.is_available() else False
    torch.autograd.set_detect_anomaly(True)
    opt.logger.info("CUDA: {0}".format( cuda))
    if cuda:
        dtype = torch.cuda.FloatTensor
        opt.logger.info(torch.cuda.get_device_name(0))
    else:
        dtype = torch.FloatTensor

    DST = "./"
    DST = ""
    PATH = os.path.join(DST, model_name)
    SAVEPATH = PATH.split('.')[-1]
    SAVEDIR = ''.join(PATH.split('.')[:-1]) + '_'
    batch_size = opt.batch_size
    # _subset = ['10', '1', '3', '5', '9']
    if not subset:
        _subset = ["10", "1", "7", "8", "9"]
        #_subset = ["1", "3", "5", "7", "9"]
        opt.logger.info('Train: {0}'.format( _subset))
        subset = [i + "_" for i in _subset]
    else:
        subset = [i + "_" for i in subset]
    dataset = RENOIR_Dataset(
        img_dir=os.path.join(
            opt.train
        ),
        transform=transforms.Compose([standardize(normalize=False), ToTensor()]),
        subset=subset,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True   , pin_memory=True, num_workers=4, drop_last=True
    )

    width = args.width
    supporting_matrix(opt)
    total_epoch = epoch
    opt.logger.info("Dataset: {0}".format( len(dataset)))
    gtv = DeepGTV(
        width=args.width,
        prox_iter=1,
        u_max=10,
        u_min=0.5,
        lambda_min=0.5,
        lambda_max=1e9,
        cuda=cuda,
        opt=opt,
    )
    if args.stack:
        gtv.load(p1=args.stack, p2=args.stack)
        opt.logger.info("Stacked from ", args.stack)
    else:
        opt.logger.info("Train DGTV from scratch")

    if cont:
        gtv.load_state_dict(torch.load(cont))
        opt.logger.info("LOAD PREVIOUS DGTV:", cont)
    if cuda:
        gtv.gtv1.cuda()
        #gtv.gtv2.cuda()
        gtv.cuda()
    criterion = nn.MSELoss()
    
    #gtv1_params = list(filter(lambda kv: 'gtv1' in kv[0] , gtv.named_parameters()))
    #gtv1_params = [i[1] for i in gtv1_params ]
    cnnf_params = list(filter(lambda kv: 'cnnf' in kv[0], gtv.named_parameters()))
    cnnf_params = [i[1] for i in cnnf_params]



    optimizer = optim.SGD([
                {'params': cnnf_params, 'lr':opt.lr},
             ], lr=opt.lr, momentum=opt.momentum)

    optimizer = optim.SGD(gtv.parameters(), lr=opt.lr, momentum=opt.momentum)
    if cont:
        try:
            optimizer.load_state_dict(torch.load(cont+'optim'))
            opt.logger.info("LOAD PREVIOUS OPTIMIZER:", cont+'optim')
        except Exception:
            opt.logger.info("Using new optimizer")
    current_lr = opt.lr

    hist = list()
    losshist = list()
    tstart = time.time()
    tprev=tstart
    opt._print()
    pickle.dump(opt, open( "dopt", "wb" ))
    ld = len(dataset)
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        # running_loss_inside = 0.0
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):  # start index at 0
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["nimg"][:, : opt.channels, :, :].float().type(dtype)
            labels = data["rimg"][:, : opt.channels, :, :].float().type(dtype)
            # zero the parameter gradients

            optimizer.zero_grad()
            #for op in optimizer:
            #    op.zero_grad()
            # forward + backward + optimize
            outputs = gtv(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_value_(cnnf_params, 1e1)
            #torch.nn.utils.clip_grad_value_(cnny_params, 1)
            #torch.nn.utils.clip_grad_value_(cnnu_params, 1e1)

            optimizer.step()
            #optimizer[i%3].step()
            running_loss += loss.item()

            if epoch==0 and (i+1)%80==0:
                g = gtv.gtv1
                with torch.no_grad():
                    P1 = g(inputs, debug=1, Tmod= 5)
                if opt.ver: # experimental version
                    opt.logger.info("\tCNNF stats: {0:.5f}".format( g.cnnf.layer[0].weight.grad.median().item()))
                else:
                    opt.logger.info("\tCNNF stats: {0:.5f}".format( g.cnnf.layer1[0].weight.grad.mean().item()))
                opt.logger.info("\tCNNU grads: {0:.5f}".format( g.cnnu.layer[0].weight.grad.mean().item()))
                opt.logger.info("\tCNNS grads: {0:.5f}".format( g.cnns.layer[0].weight.grad.mean().item()))
                with torch.no_grad():
                    us = g.cnnu(inputs)
                    opt.logger.info("\tCNNU stats: max {0:.5f} mean {1:.5f} min {2:.5f}".format( us.max().item(),  us.mean().item(),us.min().item()))
                    us = g.cnns(inputs)
                    opt.logger.info("\tCNNS stats: max {0:.5f} mean {1:.5f} min {2:.5f}".format( us.max().item(),  us.mean().item(),us.min().item()))
                with torch.no_grad():
                    P2 = g(P1, debug=1, Tmod= 5)
                with torch.no_grad():
                    P3 = g(P2, debug=1, Tmod= 5)


        tnow = time.time()
        opt.logger.info(
            '[{0}] \x1b[31mLOSS\x1b[0m: {1:.3f}, time elapsed: {2:.1f} secs, epoch time: {3:.1f} secs'.format(
                epoch + 1, running_loss / (ld*(i+1)), tnow - tstart, tnow-tprev
            )
        )
        tprev=tnow
        

        if ((epoch + 1) % 1 == 0) or (epoch + 1) == total_epoch:
            g = gtv.gtv1
            with torch.no_grad():
                histW = g(inputs, debug=1, Tmod= 5)
            if opt.ver: # experimental version
                opt.logger.info("\tCNNF stats: {0:.5f}".format( g.cnnf.layer[0].weight.grad.median().item()))
            else:
                opt.logger.info("\tCNNF stats: {0:.5f}".format( g.cnnf.layer1[0].weight.grad.mean().item()))
            opt.logger.info("\tCNNU grads: {0:.5f}".format( g.cnnu.layer[0].weight.grad.mean().item()))
            opt.logger.info("\tCNNS grads: {0:.5f}".format( g.cnns.layer[0].weight.grad.mean().item()))

            with torch.no_grad():
                us = g.cnnu(inputs[:10])
                opt.logger.info("\tCNNU stats: max {0:.5f} mean {1:.5f} min {2:.5f}".format( us.max().item(),  us.mean().item(),us.min().item()))
                us = g.cnns(inputs)
                opt.logger.info("\tCNNS stats: max {0:.5f} mean {1:.5f} min {2:.5f}".format( us.max().item(),  us.mean().item(),us.min().item()))

            with torch.no_grad():
                P2 = g(P1, debug=1, Tmod= 5)
            with torch.no_grad():
                P3 = g(P2, debug=1, Tmod= 5)


            opt.logger.info("\tsave @ epoch {0}".format( epoch + 1))
            torch.save(gtv.state_dict(), SAVEDIR + str(epoch) +'.'+SAVEPATH)
            torch.save(optimizer.state_dict(), SAVEDIR + str(epoch)+'.'+SAVEPATH + "optim")


        #scheduler.step() 
        losshist.append(running_loss / (ld*(i+1)))
        torch.save(gtv.state_dict(), SAVEDIR + str(epoch) +'.'+SAVEPATH)
    torch.save(optimizer.state_dict(), SAVEDIR + str(epoch)+'.'+SAVEPATH + "optim")
           
    opt.logger.info("Total running time: {0:.3f}".format(time.time() - tstart))
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    cumsum_vec = np.cumsum(np.insert(losshist, 0, 0))
    window_width = 30
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    ax.plot(ma_vec)
    fig.savefig("loss.png")

opt = OPT(batch_size = 50,  channels=3,  u=50, lr=8e-6, momentum=0.9, u_max=65, u_min=50, cuda=True if torch.cuda.is_available() else False)
#batch_size = 50, admm_iter=4, prox_iter=3, delta=.1, channels=3, eta=.3, u=50, lr=8e-6, momentum=0.9, u_max=65, u_min=50)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-m", "--model"
    )
    parser.add_argument(
        "-c", "--cont", default=None
    )
    parser.add_argument(
        "--batch", default=64
    )
    parser.add_argument(
        "--lr", default=8e-6
    )
    parser.add_argument(
        "--delta", default=0.05
    )
    parser.add_argument(
        "--eta", default=0.05, type=float
    )
    parser.add_argument(
        "--admm_iter", default=4
    )
    parser.add_argument(
        "--epoch", default=200
    )
    parser.add_argument(
        "--umax", default=65, type=float
    )
    parser.add_argument(
        "--umin", default=50, type=float
    )
    parser.add_argument(
        "--seed", default=0, type=float
    )
    parser.add_argument(
            "--train", default='gauss_batch')
    parser.add_argument(
            "--stack", default=None)
    parser.add_argument(
        "--width", default=36, type=int
    )

    args = parser.parse_args()
    if args.model:
        model_name = args.model
    else:
        model_name='DGTV.pkl'
    opt.batch_size = int(args.batch) 
    opt.lr = float(args.lr)
    opt.admm_iter = int(args.admm_iter)
    opt.delta = float(args.delta)
    opt.eta=args.eta
    opt.u_min=args.umin
    opt.u_max=args.umax
    opt.ver=True
    opt.train=args.train
    opt.width=args.width
    torch.manual_seed(args.seed)
    logging.basicConfig(filename='log/dgtv_train_{0}.log'.format(time.strftime("%Y-%m-%d-%H%M")),
                            filemode='a',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.NOTSET)

    logger = logging.getLogger('root')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    opt.logger=logger
    logger.info("Train DGTV")
    main(seed=1, model_name=model_name, cont=args.cont, epoch=int(args.epoch), subset=['1', '3', '5', '7', '9'])
