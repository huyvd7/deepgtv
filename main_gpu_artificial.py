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

cuda = True if torch.cuda.is_available() else False
if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


class cnnf(nn.Module):
    """
    CNN F of GLR
    """

    def __init__(self):
        super(cnnf, self).__init__()
        self.layer1 = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(opt.channels, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
        )
        self.layer2a = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        self.layer3a = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # self.maxpool
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # DECONVO

        self.deconvo1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
        )

        # CONCAT with output of layer2
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
        )
        # DECONVO
        self.deconvo2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
        )

        # CONCAT with output of layer1
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            # nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1),
        )
        self.relu = nn.LeakyReLU(0.05)  # nn.ReLU()

    def forward(self, x):
        outl1 = self.layer1(x)
        outl2 = self.layer2a(outl1)
        outl2 = self.maxpool(outl2)
        outl2 = self.layer2(outl2)
        outl3 = self.layer3a(outl2)
        outl3 = self.maxpool(outl3)
        outl3 = self.layer3(outl3)
        outl3 = self.deconvo1(outl3)
        outl3 = torch.cat((outl3, outl2), dim=1)
        outl4 = self.layer4(outl3)
        outl4 = self.deconvo2(outl4)
        outl4 = torch.cat((outl4, outl1), dim=1)
        del outl1, outl2, outl3
        out = self.layer5(outl4)
        return out


class cnnu(nn.Module):
    """
    CNNU of GLR
    """

    def __init__(self, u_min=1e-3):
        super(cnnu, self).__init__()
        self.layer = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(opt.channels, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

        self.u_min = u_min
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 32, 1 * 1 * 32),
            nn.Linear(1 * 1 * 32, 1),
            # nn.ReLU()
            nn.LeakyReLU(0.05),
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


class cnny(nn.Module):
    """
    CNN Y of GLR
    """

    def __init__(self):
        super(cnny, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(opt.channels, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, opt.channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out = identity + out
        del identity
        return out


class RENOIR_Dataset(Dataset):
    """
    Dataset loader
    """

    def __init__(self, img_dir, transform=None, subset=None):
        """
        Args:
            img_dir (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.npath = os.path.join(img_dir, "noisy")
        self.rpath = os.path.join(img_dir, "ref")
        self.subset = subset
        self.nimg_name = sorted(os.listdir(self.npath))
        self.rimg_name = sorted(os.listdir(self.rpath))
        self.nimg_name = [
            i
            for i in self.nimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp", "tif"]
        ]

        self.rimg_name = [
            i
            for i in self.rimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp"]
        ]

        if self.subset:
            nimg_name = list()
            rimg_name = list()
            for i in range(len(self.nimg_name)):
                for j in self.subset:
                    if j in self.nimg_name[i]:
                        nimg_name.append(self.nimg_name[i])
                        # if j in self.rimg_name[i]:
                        rimg_name.append(self.rimg_name[i])
            self.nimg_name = sorted(nimg_name)
            self.rimg_name = sorted(rimg_name)

        self.transform = transform

    def __len__(self):
        return len(self.nimg_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        uid = np.random.randint(0, 8)
        nimg_name = os.path.join(self.npath, self.nimg_name[idx])
        nimg = cv2.imread(nimg_name)
        nimg = data_aug(nimg, uid)
        rimg_name = os.path.join(self.rpath, self.rimg_name[idx])
        rimg = cv2.imread(rimg_name)
        rimg = data_aug(rimg, uid)

        sample = {"nimg": nimg, "rimg": rimg}

        if self.transform:
            sample = self.transform(sample)

        return sample


class standardize(object):
    """Convert opencv BGR to RGB order. Scale the image with a ratio"""

    def __init__(self, scale=None, w=None, normalize=None):
        """
        Args:
        scale (float): resize height and width of samples to scale*width and scale*height
        width (float): resize height and width of samples to width x width. Only works if "scale" is not specified
        """
        self.scale = scale
        self.w = w
        self.normalize = normalize

    def __call__(self, sample):
        nimg, rimg = sample["nimg"], sample["rimg"]
        if self.scale:
            nimg = cv2.resize(nimg, (0, 0), fx=self.scale, fy=self.scale)
            rimg = cv2.resize(rimg, (0, 0), fx=self.scale, fy=self.scale)
        else:
            if self.w:
                nimg = cv2.resize(nimg, (self.w, self.w))
                rimg = cv2.resize(rimg, (self.w, self.w))
        if self.normalize:
            nimg = cv2.resize(nimg, (0, 0), fx=1, fy=1)
            rimg = cv2.resize(rimg, (0, 0), fx=1, fy=1)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        if self.normalize:
            nimg = nimg / 255
            rimg = rimg / 255
        return {"nimg": nimg, "rimg": rimg}


class gaussian_noise_(object):
    def __init__(self, stddev, mean):
        self.stddev = stddev
        self.mean = mean

    def __call__(self, sample):
        nimg, rimg = sample["rimg"], sample["rimg"]
        noise = Variable(nimg.data.new(nimg.size()).normal_(self.mean, self.stddev))
        nimg = nimg + noise
        nimg = _norm(nimg, 0, 255)
        return {"nimg": nimg, "rimg": rimg}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Swap color axis from H x W x C (numpy) to C x H x W (torch)
        """
        nimg, rimg = sample["nimg"], sample["rimg"]
        nimg = nimg.transpose((2, 0, 1))
        rimg = rimg.transpose((2, 0, 1))
        return {
            "nimg": torch.from_numpy(nimg).type(dtype),
            "rimg": torch.from_numpy(rimg).type(dtype),
        }

def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def connected_adjacency(image, connect=8, patch_size=(1, 1)):
    """
    Construct 8-connected pixels base graph (0 for not connected, 1 for connected)
    """
    r, c = image.shape[:2]
    r = int(r / patch_size[0])
    c = int(c / patch_size[1])

    if connect == "4":
        # constructed from 2 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c - 1), [0]), r)[:-1]
        d2 = np.ones(c * (r - 1))
        upper_diags = ss.diags([d1, d2], [1, c])
        return upper_diags + upper_diags.T

    elif connect == "8":
        # constructed from 4 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c - 1), [0]), r)[:-1]
        d2 = np.append([0], d1[: c * (r - 1)])
        d3 = np.ones(c * (r - 1))
        d4 = d2[1:-1]
        upper_diags = ss.diags([d1, d2, d3, d4], [1, c - 1, c, c + 1])
        return upper_diags + upper_diags.T


def get_w(ij, F):
    """
    Compute weights for node i and node j using exemplars F
    """
    W = w(
        (
            (
                F.unsqueeze(-1).repeat(1, 1, 1, 4)
                - F.unsqueeze(-1).repeat(1, 1, 1, 4).permute(0, 1, 3, 2)
            )
            ** 2
        ).sum(axis=1)
    )

    return W.type(dtype)


def gauss(d, epsilon=1):
    """
    Compute (3)
    """

    return torch.exp(-d / (2 * epsilon ** 2))


def graph_construction(opt, F):
    """
    Construct Laplacian matrix
    """
    #     Fs = F.unsqueeze(-1).repeat(1, 1, 1, F.shape[-1])
    Fs = (opt.H.matmul(F) ** 2).requires_grad_(True)
    W = gauss(Fs.sum(axis=1)).requires_grad_(True)
    return W


def weights_init_normal(m):
    """
    Initialize weights of convolutional layers
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class OPT:
    def __init__(
        self,
        batch_size=100,
        width=36,
        connectivity="8",
        admm_iter=1,
        prox_iter=1,
        delta=1,
        channels=3,
        eta=0.1,
        u=1,
        u_max=100,
        u_min=10,
        lr=1e-4,
        momentum=0.99,
    ):
        self.batch_size = batch_size
        self.width = width
        self.edges = 0
        self.nodes = width ** 2
        self.I = None
        self.pairs = None
        self.H = None
        self.connectivity = connectivity
        self.admm_iter = admm_iter
        self.prox_iter = prox_iter
        self.channels = channels
        self.eta = eta
        self.u = u
        self.lr = lr
        self.delta = delta
        self.momentum = momentum
        self.u_max = u_max
        self.u_min = u_min

    def _print(self):
        print(
            "batch_size =",
            self.batch_size,
            ", width =",
            self.width,
            ", admm_iter =",
            self.admm_iter,
            ", prox_iter =",
            self.prox_iter,
            ", delta =",
            self.delta,
            ", channels =",
            self.channels,
            ", eta =",
            self.eta,
            ", u =",
            self.u,
            ", lr =",
            self.lr,
            ", momentum =",
            self.momentum,
        )


class GTV(nn.Module):
    """
    GLR network
    """

    def __init__(
        self,
        width=36,
        prox_iter=5,
        u_min=1e-3,
        u_max=1,
        lambda_min=1e-9,
        lambda_max=1e9,
        cuda=False,
        opt=None,
    ):
        super(GTV, self).__init__()
        self.cnnf = cnnf()

        self.opt = opt
        self.wt = width
        self.width = width
        self.cnnu = cnnu(u_min=u_min)

        self.cnny = cnny()

        if cuda:
            self.cnnf.cuda()
            self.cnnu.cuda()
            self.cnny.cuda()

        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.cnnf.apply(weights_init_normal)
        self.cnny.apply(weights_init_normal)
        self.cnnu.apply(weights_init_normal)

    def forward(self, xf, debug=False, Tmod=False):  # gtvforward
        #u = opt.u
        u = self.cnnu.forward(xf)
        u_max = opt.u_max
        u_min = opt.u_min
        u = torch.clamp(u, u_min, u_max)
        u = u.unsqueeze(1).unsqueeze(1)
        if debug:
            self.u=u.clone()
        x = xf.view(xf.shape[0], xf.shape[1], opt.width ** 2, 1).requires_grad_(True)
        z = opt.H.matmul(x).requires_grad_(True)

        ###################
        E = self.cnnf.forward(xf)
        Fs = (
            opt.H.matmul(E.view(E.shape[0], E.shape[1], opt.width ** 2, 1)) ** 2
        ).requires_grad_(True)
        w = torch.exp(-(Fs.sum(axis=1)) / (2 * (1 ** 2))).requires_grad_(True)
        ###################
        if debug:
            print("\tWEIGHT SUM", w[0, :, :].sum().data)
            hist = list()
        w = w.unsqueeze(1).repeat(1, opt.channels, 1, 1)
        T = opt.admm_iter
        P = opt.prox_iter
        if debug:
            if Tmod:
                T = Tmod
        delta = opt.delta
        eta = opt.eta
        lagrange = opt.lagrange.requires_grad_(True)

        Y = self.cnny.forward(xf).squeeze(0)
        y = Y.view(xf.shape[0], xf.shape[1], opt.width ** 2, 1).requires_grad_(True)
        I = opt.I.requires_grad_(True)
        H = opt.H.requires_grad_(True)
        D = (
            torch.inverse(2 * opt.I + delta * (opt.H.T.mm(H)))
            .type(dtype)
            .requires_grad_(True)
        )
        for i in range(T):
            # STEP 1
            xhat = D.matmul(
                2 * y - H.T.matmul(lagrange) + delta * H.T.matmul(z)
            ).requires_grad_(True)
            if i == 0:
                z = opt.H.matmul(xhat).requires_grad_(True)
            
            # STEP 2
            for j in range(P):
                grad = (delta * z - lagrange - delta * H.matmul(xhat)).requires_grad_(
                    True
                )
                z = proximal_gradient_descent(
                    x=z, grad=grad, w=w, u=u, eta=eta, debug=debug
                ).requires_grad_(True)

            # STEP 3
            lagrange = (lagrange + delta * (H.matmul(xhat) - z)).requires_grad_(True)
            if debug:
                l = (
                    (
                        (y - xhat).permute(0, 1, 3, 2).matmul(y - xhat)
                        + (u * w * z.abs()).sum(axis=[1, 2, 3])
                    )
                    + lagrange.permute(0, 1, 3, 2).matmul(H.matmul(xhat) - z)
                    + (delta / 2)
                    * (H.matmul(xhat) - z)
                    .permute(0, 1, 3, 2)
                    .matmul(H.matmul(xhat) - z)
                )
                hist.append(l[:, 0, :, :])
    
        # xhat = D.matmul(2*y - H.T.matmul(lagrange) + delta*H.T.matmul(z)).requires_grad_(True)
        if debug:
            print("min - max xhat: ", xhat.min().data, xhat.max().data)
            hist = [h.flatten() for h in hist]
            return hist
        return xhat.view(xhat.shape[0], opt.channels, opt.width, opt.width)

    def predict(self, xf):
        pass


def supporting_matrix(opt):
    width = opt.width

    pixel_indices = [i for i in range(width * width)]
    pixel_indices = np.reshape(pixel_indices, (width, width))
    A = connected_adjacency(pixel_indices, connect=opt.connectivity)
    A_pair = np.asarray(np.where(A.toarray() == 1)).T
    A_pair = np.unique(np.sort(A_pair, axis=1), axis=0)

    opt.edges = A_pair.shape[0]
    H_dim0 = opt.edges
    H_dim1 = width ** 2
    # unique_A_pair = np.unique(np.sort(A_pair, axis=1), axis=0)

    I = torch.eye(width ** 2, width ** 2).type(dtype)
    lagrange = torch.zeros(opt.edges, 1).type(dtype)
    A = torch.zeros(width ** 2, width ** 2).type(dtype)
    H = torch.zeros(H_dim0, H_dim1).type(dtype)
    for e, p in enumerate(A_pair):
        H[e, p[0]] = 1
        H[e, p[1]] = -1
        A[p[0], p[1]] = 1

    opt.I = I.type(dtype).requires_grad_(True)
    opt.pairs = A_pair
    opt.H = H.type(dtype).requires_grad_(True)
    opt.connectivity_full = A.requires_grad_(True)
    opt.connectivity_idx = torch.where(A > 0)
    opt.lagrange = lagrange.requires_grad_(True)
    delta = 1
    # opt.D = torch.inverse(2*opt.I + delta*(opt.H.T.mm(H))).type(dtype)


def proximal_gradient_descent(x, grad, w, u=1, eta=1, debug=False):
    v = x - eta * grad
    masks1 = ((v.abs() - (eta * w * u).abs()) > 0).type(dtype).requires_grad_(True)
    masks2 = ((v.abs() - (eta * w * u).abs()) <= 0).type(dtype).requires_grad_(True)
    v = v - masks1 * eta * w * u * torch.sign(v)
    v = v - masks2 * v
    return v


def _norm(x, newmin, newmax):
    return (x - x.min()) * (newmax - newmin) / (x.max() - x.min() + 1e-8) + newmin


def printmax(x):
    print(x.max().data[0])


def printmean(x):
    print(x.mean().data[0])


def printall(x):
    print(x.median().data, x.max().data, x.min().data)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def printfull(x):
    # print(check_symmetric(x[0,0,:].cpu().detach().numpy()))
    print(x.median().data[0], x.max().data[0], x.min().data[0], end="\r")
    if debug == 1:
        global xd
        xd = x.clone()
        return x
# STD = 20
# opt = OPT(batch_size = 50, admm_iter=2, prox_iter=3, delta=.1, channels=3, eta=.3, u=50, lr=1e-5, momentum=0.9, u_max=75, u_min=25)
# STD = 50

def main(seed, model_name, cont=None, optim_name=None, subset=None, epoch=100):
    debug = 0

    xd = None
    cuda = True if torch.cuda.is_available() else False
    torch.autograd.set_detect_anomaly(True)
    print("CUDA: ", cuda)
    if cuda:
        dtype = torch.cuda.FloatTensor
        print(torch.cuda.get_device_name(0))
    else:
        dtype = torch.FloatTensor

    DST = "./"
    DST = ""
    PATH = os.path.join(DST, model_name)
    SAVEPATH = PATH
    batch_size = opt.batch_size
    # _subset = ['10', '1', '3', '5', '9']
    if not subset:
        _subset = ["10", "1", "7", "8", "9"]
        #_subset = ["1", "3", "5", "7", "9"]
        print('Train: ', _subset)
        subset = [i + "_" for i in _subset]
    else:
        subset = [i + "_" for i in subset]
    dataset = RENOIR_Dataset(
        img_dir=os.path.join(
            "C:\\Users\\HUYVU\\AppData\\Local\\Packages\\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\\LocalState\\rootfs\\home\\huyvu\\gauss_batch"
        ),
        transform=transforms.Compose([standardize(normalize=False), ToTensor()]),
        subset=subset,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True  # , pin_memory=True
    )

    width = 36
    supporting_matrix(opt)
    total_epoch = epoch
    print("Dataset: ", len(dataset))
    gtv = GTV(
        width=36,
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
        print("LOAD PREVIOUS GTV:", cont)
    if cuda:
        gtv.cuda()
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(gtv.parameters(), lr=opt.lr, momentum=opt.momentum)
    
    cnny_params = list(filter(lambda kv: 'cnny' in kv[0] , gtv.named_parameters()))
    cnny_params = [i[1] for i in cnny_params]
    cnnf_params = list(filter(lambda kv: 'cnnf' in kv[0], gtv.named_parameters()))
    cnnf_params = [i[1] for i in cnnf_params]
    cnnu_params = list(filter(lambda kv: 'cnnu' in kv[0], gtv.named_parameters()))
    cnnu_params = [i[1] for i in cnnu_params ]
    optimizer = optim.SGD([
                {'params': cnny_params, 'lr':opt.lr},
                 {'params': cnnf_params , 'lr': opt.lr*50},
                 {'params': cnnu_params , 'lr': opt.lr*20}
             ], lr=opt.lr, momentum=opt.momentum)
    #optimizer = optim.SGD(gtv.parameters(), lr=opt.lr, momentum=opt.momentum)
    if cont:
        optimizer.load_state_dict(torch.load(cont+'optim'))
        print("LOAD PREVIOUS OPTIMIZER:", cont+'optim')
    current_lr = opt.lr

    hist = list()
    losshist = list()
    tstart = time.time()
    opt._print()
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
            # forward + backward + optimize
            outputs = gtv(inputs, debug=0)
            loss = criterion(outputs, labels)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(gtv.parameters(), 1e5)
            torch.nn.utils.clip_grad_norm_(cnnf_params, 1e2)
            torch.nn.utils.clip_grad_norm_(cnny_params, 1)
            torch.nn.utils.clip_grad_norm_(cnnu_params, 1e1)

            optimizer.step()
            running_loss += loss.item()
        print(
            time.ctime(),
            '[{0}] \x1b[31mLOSS\x1b[0m: {1:.3f}, time elapsed: {2:.1f} secs'.format(
                epoch + 1, running_loss / ld, time.time() - tstart
            ),
        )
        

        if ((epoch + 1) % 1 == 0) or (epoch + 1) == total_epoch:
            with torch.no_grad():
                histW = gtv(inputs[:1, :, :, :], debug=1, Tmod=opt.admm_iter + 4)
            print("\tCNNF stats: ", gtv.cnnf.layer1[0].weight.grad.mean())
            print("\tCNNU mean: ", gtv.u.mean().data)
            print("\tCNNU grads: ", gtv.cnnu.layer[0].weight.grad.mean())
            pmax = list()
            for p in gtv.parameters():
                pmax.append(p.grad.max())
            print("\tmax gradients", max(pmax))

            print("\tsave @ epoch ", epoch + 1)
            torch.save(gtv.state_dict(), SAVEPATH)
            torch.save(optimizer.state_dict(), SAVEPATH + "optim")
            histW = [h.cpu().detach().numpy()[0] for h in histW]
            print("\t", np.argmin(histW), min(histW), histW)

        #scheduler.step() 
        losshist.append(running_loss / ld)
        if (epoch+1) in [100000]:
            print("CHANGE LR")
            current_lr /= 5
            optimizer = optim.SGD(gtv.parameters(), lr=current_lr, momentum=opt.momentum)
            #optimizer = optim.SGD([
            #        {'params': base_params},
            #        {'params': cnny_params , 'lr': current_lr*20}], lr=current_lr, momentum=opt.momentum)
    torch.save(gtv.state_dict(), SAVEPATH)
    torch.save(optimizer.state_dict(), SAVEPATH + "optim")
    print("Total running time: {0:.3f}".format(time.time() - tstart))
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    cumsum_vec = np.cumsum(np.insert(losshist, 0, 0))
    window_width = 30
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    ax.plot(ma_vec)
    fig.savefig("loss.png")

opt = OPT(batch_size = 50, admm_iter=4, prox_iter=3, delta=.9, channels=3, eta=.3, u=25, lr=8e-6, momentum=0.9, u_max=1e3, u_min=1e-1)

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
        "--lr", default=8e-6
    )
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
    main(seed=1, model_name=model_name, cont=cont, epoch=200, subset=['1', '3', '5', '7', '9'])
