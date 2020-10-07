import scipy.sparse as ss
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
import pickle
import logging
import sys
logging.basicConfig(filename='main_gpu_artificial_{0}.log'.format(time.strftime("%Y-%m-%d-%H%M")),
                            filemode='a',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.NOTSET)

logging.info("Running DGTV evaluation")

logger = logging.getLogger('root')
logger.addHandler(logging.StreamHandler(sys.stdout))
def main(seed, model_name, cont=None, optim_name=None, subset=None, epoch=100, args=None):
    debug = 0
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
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True
    )

    width = args.width
    supporting_matrix(opt)
    total_epoch = epoch
    opt.logger.info("Dataset: {0}".format( len(dataset)))
    gtv = GTV(
        width=args.width,
        prox_iter=1,
        u_max=10,
        u_min=0.5,
        lambda_min=0.5,
        lambda_max=1e9,
        cuda=cuda,
        opt=opt,
    )
    if cont:
        gtv.load_state_dict(torch.load(cont))
        opt.logger.info("LOAD PREVIOUS GTV:", cont)
    if cuda:
        gtv.cuda()
    criterion = nn.MSELoss()
    
    cnnf_params = list(filter(lambda kv: 'cnnf' in kv[0], gtv.named_parameters()))
    cnnf_params = [i[1] for i in cnnf_params]
    cnnu_params = list(filter(lambda kv: 'cnnu' in kv[0], gtv.named_parameters()))
    cnnu_params = [i[1] for i in cnnu_params ]
    optimizer = optim.SGD([
                 {'params': cnnf_params , 'lr': opt.lr},
                 {'params': cnnu_params , 'lr': opt.lr}
             ], lr=opt.lr, momentum=opt.momentum)
    optimizer = optim.SGD(gtv.parameters(), lr=opt.lr, momentum=opt.momentum)


    if cont:
        optimizer.load_state_dict(torch.load(cont+'optim'))
        opt.logger.info("LOAD PREVIOUS OPTIMIZER:", cont+'optim')
    current_lr = opt.lr

    hist = list()
    losshist = list()
    tstart = time.time()
    tprev=tstart

    opt._print()
    pickle.dump(opt, open( "opt", "wb" ))
    ld = len(dataset)
    

    #scaler = torch.cuda.amp.GradScaler()
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        # running_loss_inside = 0.0
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):  # start index at 0
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["nimg"][:, : opt.channels, :, :].float().type(dtype)
            labels = data["rimg"][:, : opt.channels, :, :].float().type(dtype)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            #outputs = gtv.forward_approx(inputs, debug=0)
            outputs = gtv(inputs, debug=0)
            loss = criterion(outputs, labels)
            #with torch.cuda.amp.autocast():
            #    loss = criterion(outputs, labels)
            #scaler.scale(loss).backward()

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(cnnf_params, 1e1)
            #torch.nn.utils.clip_grad_norm_(cnnu_params, 1e1)
            torch.nn.utils.clip_grad_norm_(gtv.parameters(), 5e1)


            #scaler.step(optimizer)
            optimizer.step()
            #scaler.update()
            running_loss += loss.item()

            if epoch==0 and (i+1)%80==0:
                with torch.no_grad():
                    histW = gtv(inputs, debug=1, Tmod=5)
                if opt.ver: # experimental version
                    opt.logger.info("\tCNNF stats: {0:.5f}".format( gtv.cnnf.layer[0].weight.grad.median().item()))
                else:
                    opt.logger.info("\tCNNF stats: {0:.5f}".format( gtv.cnnf.layer1[0].weight.grad.mean().item()))
                opt.logger.info("\tCNNU grads: {0:.5f}".format( gtv.cnnu.layer[0].weight.grad.mean().item()))
                opt.logger.info("\tCNNS grads: {0:.5f}".format( gtv.cnns.layer[0].weight.grad.mean().item()))

                with torch.no_grad():
                    us = gtv.cnnu(inputs)
                    opt.logger.info("\tCNNU stats: max {0:.5f} mean {1:.5f} min {2:.5f}".format( us.max().item(),  us.mean().item(),us.min().item()))
                    us = gtv.cnns(inputs)
                    opt.logger.info("\tCNNS stats: max {0:.5f} mean {1:.5f} min {2:.5f}".format( us.max().item(),  us.mean().item(),us.min().item()))


        tnow = time.time()
        opt.logger.info(
            '[{0}] \x1b[31mLOSS\x1b[0m: {1:.3f}, time elapsed: {2:.1f} secs, epoch time: {3:.1f} secs'.format(
                epoch + 1, running_loss / (ld*(i+1)), tnow - tstart, tnow-tprev
            )
        )
        tprev=tnow
        

        if ((epoch + 1) % 1 == 0) or (epoch + 1) == total_epoch:
            with torch.no_grad():
                histW = gtv(inputs, debug=1, Tmod= 5)
            if opt.ver: # experimental version
                opt.logger.info("\tCNNF stats: {0:.5f}".format( gtv.cnnf.layer[0].weight.grad.median().item()))
            else:
                opt.logger.info("\tCNNF stats: {0:.5f}".format( gtv.cnnf.layer1[0].weight.grad.mean().item()))
                opt.logger.info("\tCNNU grads: {0:.5f}".format( gtv.cnnu.layer[0].weight.grad.mean().item()))
                opt.logger.info("\tCNNS grads: {0:.5f}".format( gtv.cnns.layer[0].weight.grad.mean().item()))

            pmax = list()
            for p in gtv.parameters():
                pmax.append(p.grad.max())
            opt.logger.info("\tmax gradients {0}".format( max(pmax)))
            with torch.no_grad():
                us = gtv.cnnu(inputs)
                opt.logger.info("\tCNNU stats: max {0:.5f} mean {1:.5f} min {2:.5f}".format( us.max().item(),  us.mean().item(),us.min().item()))
                us = gtv.cnns(inputs)
                opt.logger.info("\tCNNS stats: max {0:.5f} mean {1:.5f} min {2:.5f}".format( us.max().item(),  us.mean().item(),us.min().item()))


            opt.logger.info("\tsave @ epoch {0}".format( epoch + 1))
            torch.save(gtv.state_dict(), SAVEDIR + str(epoch) +'.'+SAVEPATH)
            torch.save(optimizer.state_dict(), SAVEDIR + str(epoch)+'.'+SAVEPATH + "optim")

        losshist.append(running_loss / (ld*(i+1)))
    torch.save(gtv.state_dict(), SAVEDIR + str(epoch) +'.'+SAVEPATH)
    torch.save(optimizer.state_dict(), SAVEDIR + str(epoch)+'.'+SAVEPATH + "optim")
           
    opt.logger.info("Total running time: {0:.3f}".format(time.time() - tstart))
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    cumsum_vec = np.cumsum(np.insert(losshist, 0, 0))
    window_width = 30
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    ax.plot(ma_vec)
    ax.set(ylim=[0, ax.get_ylim()[1] *1.05 ])
    fig.savefig("loss.png")

opt = OPT(batch_size = 50, channels=3, u=50, lr=8e-6, momentum=0.9, u_max=65, u_min=50, cuda=True if torch.cuda.is_available() else False)
#batch_size = 50, admm_iter=4, prox_iter=3, delta=.1, channels=3, eta=.3, u=50, lr=8e-6, momentum=0.9, u_max=65, u_min=50)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-m", "--model"
    )
    parser.add_argument(
        "-c", "--cont"
    )
    parser.add_argument(
        "--batch", default=64
    )
    parser.add_argument(
        "--lr", default=8e-6, type=float
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
        "--width", default=36, type=int
    )
    parser.add_argument(
            "--train", default='gauss_batch')

    args = parser.parse_args()
    if args.cont:
        cont = args.cont
    else:
        cont = None
    if args.model:
        model_name = args.model
    else:
        model_name='GTV.pkl'
    opt.batch_size = int(args.batch) 
    opt.lr = float(args.lr)
    opt.u_min=args.umin
    opt.u_max=args.umax
    opt.ver=True
    opt.train=args.train
    opt.width=args.width
    torch.manual_seed(args.seed)
    opt.logger=logger
    main(seed=1, model_name=model_name, cont=cont, epoch=int(args.epoch), subset=['1', '3', '5', '7', '9'], args=args)
