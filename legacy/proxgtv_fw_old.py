import scipy.sparse as ss
import shutil
import torch
import numpy as np
import os
import cv2
import torch.nn as nn

from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

cuda = True if torch.cuda.is_available() else False

if cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

dv = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class cnnf_2(nn.Module):
    def __init__(self, opt):
        super(cnnf_2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(opt.channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 6, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class uu(nn.Module):
    def __init__(self):
        super(uu,self).__init__()
        self.u = torch.nn.Parameter(torch.rand(1), requires_grad=True)
    def forward(self):
        return self.u

class cnnu(nn.Module):
    """
    CNNU of GLR
    """

    def __init__(self, u_min=1e-3, opt=None):
        super(cnnu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(opt.channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

        self.opt = opt
        self.u_min = u_min
        self.fc = nn.Sequential(
            nn.Linear(self.linear_input_neurons(), 1 * 1 * 32),
            nn.Linear(1 * 1 * 32, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

    def size_after_relu(self, x):
        x = self.layer(x)

        return x.size()

    def linear_input_neurons(self):
        size = self.size_after_relu(
            torch.rand(1, self.opt.channels, self.opt.width, self.opt.width)
        )
        m = 1
        for i in size:
            m *= i

        return int(m)

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
        # uid = 0
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
            nimg = nimg / 255.0
            rimg = rimg / 255.0
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
            "nimg": torch.from_numpy(nimg), 
            "rimg": torch.from_numpy(rimg)
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
        channels=3,
        u_max=100,
        u_min=10,
        lr=1e-4,
        momentum=0.99,
        ver=None,
        train="gauss_batch",
        cuda=False,
        logger=None,
        legacy=False
    ):
        self.batch_size = batch_size
        self.legacy = legacy
        self.width = width
        self.edges = 0
        self.nodes = width ** 2
        self.I = None
        self.pairs = None
        self.H = None
        self.connectivity = connectivity
        self.channels = channels
        self.lr = lr
        self.momentum = momentum
        self.u_max = u_max
        self.u_min = u_min
        self.ver = ver
        self.D = None
        self.train = train
        self.cuda = cuda
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        self.logger = logger

    def _print(self):
        self.logger.info(
            "batch_size = {0}, width = {1}, channels = {2}, u_min = {3}, u_max = {4}, lr = {5}, momentum = {6}".format(
                self.batch_size,
                self.width,
                self.channels,
                self.u_min,
                self.u_max,
                self.lr,
                self.momentum,
            )
        )


class GTV(nn.Module):
    """
    GTV network
    """

    def __init__(
        self,
        width=36,
        prox_iter=5,
        u_min=1e-3,
        u_max=1,
        cuda=False,
        opt=None,
    ):
        super(GTV, self).__init__()

        self.opt = opt
        self.logger = opt.logger
        self.wt = width
        self.width = width
        self.cnnf = cnnf_2(opt=self.opt)
        if self.opt.legacy:
            self.cnnu = cnnu(u_min=u_min, opt=self.opt)
        else:
            self.uu = uu()

        if cuda:
            self.cnnf.cuda()
        opt.logger.info("GTV created on cuda: {0}".format(cuda))
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.device = torch.device("cuda") if cuda else torch.device("cpu")
        self.cnnf.apply(weights_init_normal)
        if self.opt.legacy:
            self.cnnu.apply(weights_init_normal)

        self.support_zmax = torch.ones(1).type(self.dtype) * 0.01
        self.support_identity = torch.eye(
            self.opt.width ** 2, self.opt.width ** 2
        ).type(self.dtype)
        self.support_L = torch.ones(opt.width ** 2, 1).type(self.dtype)
        self.base_W = torch.zeros(
            self.opt.batch_size,
            self.opt.channels,
            self.opt.width ** 2,
            self.opt.width ** 2,
        ).type(self.dtype)
        self.lanczos_order = 20
        self.support_e1 = torch.zeros(self.lanczos_order, 1).type(self.dtype)
        self.support_e1[0] = 1
        self.weight_sigma = 0.01

    def forward(self, xf, debug=False, manual_debug=False):  # gtvforward
        s = self.weight_sigma
        if self.opt.legacy:
            u = self.cnnu.forward(xf)
            u = u.unsqueeze(1).unsqueeze(1)
        else:
            u=self.uu.forward()
        u_max = self.opt.u_max
        u_min = self.opt.u_min
        if debug:
            self.u = u.clone()
        u = torch.clamp(u, u_min, u_max)

        z = self.opt.H.matmul(xf.view(xf.shape[0], xf.shape[1], self.opt.width ** 2, 1))

        ###################
        E = self.cnnf.forward(xf)
        Fs = (
            self.opt.H.matmul(E.view(E.shape[0], E.shape[1], self.opt.width ** 2, 1))
            ** 2
        )

        w = torch.exp(-(Fs.sum(axis=1)) / (s ** 2))
        if debug:
            s = f"Sample WEIGHT SUM: {w[0, :, :].sum().item():.4f} || Mean Processed u: {u.mean().item():.4f}"
            self.logger.info(s)
            
        w = w.unsqueeze(1).repeat(1, self.opt.channels, 1, 1)

        W = self.base_W.clone()
        Z = W.clone()
        W[:, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]] = w.view(
            xf.shape[0], 3, -1
        )
        W[:, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]] = w.view(
            xf.shape[0], 3, -1
        )
        Z[:, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]] = torch.abs(
            z.view(xf.shape[0], 3, -1)
        )
        Z[:, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]] = torch.abs(
            z.view(xf.shape[0], 3, -1)
        )
        Z = torch.max(Z, self.support_zmax)
        L = W / Z

        L1 = L @ self.support_L
        L = torch.diag_embed(L1.squeeze(-1)) - L

        ########################
        y = xf.view(xf.shape[0], self.opt.channels, -1, 1)
        ########################

        xhat = self.qpsolve(L, u, y, self.support_identity, self.opt.channels)

        # GLR 2
        def glr(y, w, u, debug=False, return_dict=None):
            W = self.base_W.clone()
            z = self.opt.H.matmul(y)
            Z = W.clone()
            W[
                :, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]
            ] = w.view(xf.shape[0], 3, -1)
            W[
                :, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]
            ] = w.view(xf.shape[0], 3, -1)
            Z[
                :, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]
            ] = torch.abs(z.view(xf.shape[0], 3, -1))
            Z[
                :, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]
            ] = torch.abs(z.view(xf.shape[0], 3, -1))
            Z = torch.max(Z, self.support_zmax)
            L = W / Z
            L1 = L @ self.support_L
            L = torch.diag_embed(L1.squeeze(-1)) - L

            xhat = self.qpsolve(L, u, y, self.support_identity, self.opt.channels)
            return xhat
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        return xhat.view(
            xhat.shape[0], self.opt.channels, self.opt.width, self.opt.width
        )


    def forward_old(self, xf, debug=False, manual_debug=False):  # gtvforward
        s = self.weight_sigma
        if self.opt.legacy:
            u = self.cnnu.forward(xf)
            u = u.unsqueeze(1).unsqueeze(1)
        else:
            u=self.uu.forward()
        u_max = self.opt.u_max
        u_min = self.opt.u_min
        if debug:
            self.u = u.clone()
        if manual_debug:
            return_dict = {
                "Lgamma": list(),
                "z": list(),
                "gamma": list(),
                "x": list(),
                "W": list(),
                "Z": list(),
                "gtv": list(),
                "w": list(),
                "f": list(),
            }

        u = torch.clamp(u, u_min, u_max)

        z = self.opt.H.matmul(xf.view(xf.shape[0], xf.shape[1], self.opt.width ** 2, 1))

        ###################
        E = self.cnnf.forward(xf)
        if manual_debug:
            return_dict["f"].append(E)
        Fs = (
            self.opt.H.matmul(E.view(E.shape[0], E.shape[1], self.opt.width ** 2, 1))
            ** 2
        )

        w = torch.exp(-(Fs.sum(axis=1)) / (s ** 2))
        if debug:
            s = f"Sample WEIGHT SUM: {w[0, :, :].sum().item():.4f} || Mean Processed u: {u.mean().item():.4f}"
            self.logger.info(s)
            
        w = w.unsqueeze(1).repeat(1, self.opt.channels, 1, 1)

        W = self.base_W.clone()
        Z = W.clone()
        W[:, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]] = w.view(
            xf.shape[0], 3, -1
        )
        W[:, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]] = w.view(
            xf.shape[0], 3, -1
        )
        Z[:, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]] = torch.abs(
            z.view(xf.shape[0], 3, -1)
        )
        Z[:, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]] = torch.abs(
            z.view(xf.shape[0], 3, -1)
        )
        Z = torch.max(Z, self.support_zmax)
        L = W / Z

        if manual_debug:
            return_dict["gamma"].append(L)
            return_dict["w"].append(w)
            return_dict["W"].append(W)
        L1 = L @ self.support_L
        L = torch.diag_embed(L1.squeeze(-1)) - L

        ########################
        y = xf.view(xf.shape[0], self.opt.channels, -1, 1)
        ########################

        xhat = self.qpsolve(L, u, y, self.support_identity, self.opt.channels)
        if manual_debug:
            return_dict["z"].append(z)
            return_dict["Z"].append(Z)
            return_dict["x"].append(xhat)
            return_dict["Lgamma"].append(L)

        # GLR 2
        def glr(y, w, u, debug=False, return_dict=None):
            W = self.base_W.clone()
            z = self.opt.H.matmul(y)
            Z = W.clone()
            W[
                :, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]
            ] = w.view(xf.shape[0], 3, -1)
            W[
                :, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]
            ] = w.view(xf.shape[0], 3, -1)
            Z[
                :, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]
            ] = torch.abs(z.view(xf.shape[0], 3, -1))
            Z[
                :, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]
            ] = torch.abs(z.view(xf.shape[0], 3, -1))
            Z = torch.max(Z, self.support_zmax)
            L = W / Z
            if manual_debug:
                return_dict["gamma"].append(L)
                return_dict["w"].append(w)
                return_dict["W"].append(W)

            L1 = L @ self.support_L
            L = torch.diag_embed(L1.squeeze(-1)) - L

            xhat = self.qpsolve(L, u, y, self.support_identity, self.opt.channels)

            if debug:
                return_dict["z"].append(z)
                return_dict["Z"].append(Z)
                return_dict["Lgamma"].append(L)
                return_dict["x"].append(xhat)
            return xhat

        if manual_debug:
            xhat2 = glr(xhat, w, u, debug=manual_debug, return_dict=return_dict)
            xhat3 = glr(xhat2, w, u, debug=manual_debug, return_dict=return_dict)
            xhat4 = glr(xhat3, w, u, debug=manual_debug, return_dict=return_dict)
            return (
                xhat4.view(
                    xhat4.shape[0], self.opt.channels, self.opt.width, self.opt.width
                ),
                return_dict,
            )

        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)
        xhat = glr(xhat, w, u)

        return xhat.view(
            xhat.shape[0], self.opt.channels, self.opt.width, self.opt.width
        )

    def forward_approx(self, xf, debug=False, manual_debug=False):  # gtvapprox
        self.base_W = torch.zeros(
            xf.shape[0], self.opt.channels, self.opt.width ** 2, self.opt.width ** 2
        ).type(dtype)

        u = self.cnnu.forward(xf)
        u_max = self.opt.u_max
        u_min = self.opt.u_min
        if debug:
            self.u = u.clone()
        if manual_debug:
            return_dict = {
                "Lgamma": list(),
                "z": list(),
                "gamma": list(),
                "x": list(),
                "W": list(),
                "Z": list(),
                "gtv": list(),
                "w": list(),
                "f": list(),
            }

        u = torch.clamp(u, u_min, u_max)
        # u = u.unsqueeze(1).unsqueeze(1)
        u = u.unsqueeze(1)

        z = self.opt.H.matmul(xf.view(xf.shape[0], xf.shape[1], self.opt.width ** 2, 1))

        ###################
        E = self.cnnf.forward(xf)
        if manual_debug:
            return_dict["f"].append(E)
        Fs = (
            self.opt.H.matmul(E.view(E.shape[0], E.shape[1], self.opt.width ** 2, 1))
            ** 2
        )
        w = torch.exp(-(Fs.sum(axis=1)) / (self.weight_sigma ** 2))

        if manual_debug:
            # return_dict['gtv'].append((z*w).abs().sum())
            pass
        if debug:
            self.logger.info(
                "\t\x1b[31mWEIGHT SUM (1 sample)\x1b[0m {0:.6f}".format(
                    w[0, :, :].sum().item()
                )
            )
            self.logger.info(
                "\tprocessed u: Mean {0:.4f} Median {1:.4f}".format(
                    u.mean().item(), u.median().item()
                )
            )
        w = w.unsqueeze(1).repeat(1, self.opt.channels, 1, 1)

        W = self.base_W.clone()
        Z = W.clone()
        W[:, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]] = w.view(
            xf.shape[0], 3, -1
        )
        W[:, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]] = w.view(
            xf.shape[0], 3, -1
        )
        Z[:, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]] = torch.abs(
            z.view(xf.shape[0], 3, -1)
        )
        Z[:, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]] = torch.abs(
            z.view(xf.shape[0], 3, -1)
        )
        Z = torch.max(Z, self.support_zmax)
        L = W / Z

        if manual_debug:
            return_dict["gamma"].append(L)
            return_dict["w"].append(w)
            return_dict["W"].append(W)
        L1 = L @ self.support_L
        L = torch.diag_embed(L1.squeeze(-1)) - L

        ########################
        # USE CNNY
        # Y = self.cnny.forward(xf).squeeze(0)
        # y = Y.view(xf.shape[0], xf.shape[1], self.opt.width ** 2, 1)#.requires_grad_(True)
        ####
        y = xf.view(xf.shape[0], self.opt.channels, -1, 1)
        ########################

        # xhat = self.qpsolve(L, u, y, self.support_identity, self.opt.channels)
        xhat = self.lanczos_approx(
            L, self.lanczos_order, self.support_e1, y.squeeze(-1), u
        )

        if manual_debug:
            return_dict["z"].append(z)
            return_dict["Z"].append(Z)
            return_dict["x"].append(xhat)
            return_dict["Lgamma"].append(L)

        # GLR 2
        def glr(y, w, u, debug=False, return_dict=None):
            W = self.base_W.clone()
            z = self.opt.H.matmul(y)
            Z = W.clone()
            W[
                :, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]
            ] = w.view(xf.shape[0], 3, -1)
            W[
                :, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]
            ] = w.view(xf.shape[0], 3, -1)
            Z[
                :, :, self.opt.connectivity_idx[0], self.opt.connectivity_idx[1]
            ] = torch.abs(z.view(xf.shape[0], 3, -1))
            Z[
                :, :, self.opt.connectivity_idx[1], self.opt.connectivity_idx[0]
            ] = torch.abs(z.view(xf.shape[0], 3, -1))
            Z = torch.max(Z, self.support_zmax)
            L = W / Z
            if manual_debug:
                return_dict["gamma"].append(L)
                return_dict["w"].append(w)
                return_dict["W"].append(W)

            L1 = L @ self.support_L
            L = torch.diag_embed(L1.squeeze(-1)) - L

            # xhat = self.qpsolve(L, u, y, self.support_identity, self.opt.channels)
            xhat = self.lanczos_approx(
                L, self.lanczos_order, self.support_e1, y.squeeze(-1), u
            )
            if debug:
                return_dict["z"].append(z)
                return_dict["Z"].append(Z)
                return_dict["Lgamma"].append(L)
                return_dict["x"].append(xhat)
            return xhat

        if manual_debug:
            xhat2 = glr(xhat, w, u, debug=manual_debug, return_dict=return_dict)
            xhat3 = glr(xhat2, w, u, debug=manual_debug, return_dict=return_dict)
            xhat4 = glr(xhat3, w, u, debug=manual_debug, return_dict=return_dict)
            return (
                xhat4.view(
                    xhat4.shape[0], self.opt.channels, self.opt.width, self.opt.width
                ),
                return_dict,
            )

        xhat2 = glr(xhat, w, u)
        xhat3 = glr(xhat2, w, u)
        xhat4 = glr(xhat3, w, u)

        return xhat4.view(
            xhat4.shape[0], self.opt.channels, self.opt.width, self.opt.width
        )

    def predict(self, xf, change_dtype=False, new_dtype=False, layers=1):
        if change_dtype:
            self.base_W = torch.zeros(
                xf.shape[0], self.opt.channels, self.opt.width ** 2, self.opt.width ** 2
            ).type(new_dtype)
        else:
            self.base_W = torch.zeros(
                xf.shape[0], self.opt.channels, self.opt.width ** 2, self.opt.width ** 2
            ).type(dtype)
        P = self.forward(xf)
        for i in range(layers - 1):
            P = self.forward(P)
        return P

    def qpsolve(self, L, u, y, Im, channels=3):
        """
        Solve equation (2) using (6)
        """

        t = torch.inverse(Im + u * L)

        return t @ y

    def planczos(self, A, order, x):
        q = x / torch.norm(x, dim=2, keepdim=True)
        V = torch.zeros((x.shape[0], x.shape[1], x.shape[2], order), device=self.device)
        V[:, :, :, 0] = q
        q = q.unsqueeze(-1)
        H = torch.zeros((x.shape[0], x.shape[1], order + 1, order), device=self.device)
        r = A @ q
        H[:, :, 0, 0] = torch.sum(q * r, axis=[-2, -1])

        r = r - H[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1) * q
        H[:, :, 1, 0] = torch.norm(r, dim=2).squeeze(-1)

        for k in range(1, order):
            H[:, :, k - 1, k] = H[:, :, k, k - 1]
            v = q.clone()
            q = r / H[:, :, k - 1, k].unsqueeze(-1).unsqueeze(-1)

            V[:, :, :, k] = q.squeeze(-1)

            r = A @ q
            r = r - H[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1) * v

            H[:, :, k, k] = torch.sum(q * r, axis=[-2, -1])

            r = r - H[:, :, k, k].unsqueeze(-1).unsqueeze(-1) * q
            r = r - V @ (V.permute(0, 1, 3, 2) @ r)
            H[:, :, k + 1, k] = torch.norm(r, dim=2).squeeze(-1)

        return V, H[:, :, :order, :order]

    def f(x, u=0.5):
        return 1 / (1 + u * x)

    def lanczos_approx(self, L, order, e1, dx, u):
        v, H_M = self.planczos(L, order, dx)
        H_M_eval, H_M_evec = torch.symeig(H_M, eigenvectors=True)
        H_M_eval = torch.clamp(H_M_eval, 0, H_M_eval.max().item())
        fv = H_M_evec @ torch.diag_embed(f(H_M_eval, u)) @ H_M_evec.permute(0, 1, 3, 2)
        approx = torch.norm(dx, dim=2).unsqueeze(-1).unsqueeze(-1) * v @ fv @ e1
        return approx


def planczos(A, order, x):
    q = x / torch.norm(x, dim=2, keepdim=True)
    V = torch.zeros((x.shape[0], x.shape[1], x.shape[2], order), device=dv)
    V[:, :, :, 0] = q
    q = q.unsqueeze(-1)
    H = torch.zeros((x.shape[0], x.shape[1], order + 1, order), device=dv)
    r = A @ q
    H[:, :, 0, 0] = torch.sum(q * r, axis=[-2, -1])

    r = r - H[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1) * q
    H[:, :, 1, 0] = torch.norm(r, dim=2).squeeze(-1)

    for k in range(1, order):
        H[:, :, k - 1, k] = H[:, :, k, k - 1]
        v = q.clone()
        q = r / H[:, :, k - 1, k].unsqueeze(-1).unsqueeze(-1)

        V[:, :, :, k] = q.squeeze(-1)

        r = A @ q
        r = r - H[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1) * v

        H[:, :, k, k] = torch.sum(q * r, axis=[-2, -1])

        r = r - H[:, :, k, k].unsqueeze(-1).unsqueeze(-1) * q
        r = r - V @ (V.permute(0, 1, 3, 2) @ r)
        H[:, :, k + 1, k] = torch.norm(r, dim=2).squeeze(-1)

    return V, H[:, :, :order, :order]


def f(x, u=0.5):
    return 1 / (1 + u * x)


def lanczos_approx(L, order, e1, dx, u):
    v, H_M = planczos(L, order, dx)
    H_M_eval, H_M_evec = torch.symeig(H_M, eigenvectors=True)
    H_M_eval[H_M_eval < 0] = 0
    fv = H_M_evec @ torch.diag_embed(f(H_M_eval, u)) @ H_M_evec.permute(0, 1, 3, 2)
    approx = torch.norm(dx, dim=2).unsqueeze(-1).unsqueeze(-1) * v @ fv @ e1
    return approx


class DeepGTV(nn.Module):
    """
    Stack GTVs
    """

    def __init__(
        self,
        width=36,
        prox_iter=5,
        u_min=1e-3,
        u_max=1,
        cuda=False,
        opt=None
    ):
        super(DeepGTV, self).__init__()
        self.gtv1 = GTV(width=width, u_max=u_max, u_min=u_min, cuda=cuda, opt=opt,)

        self.opt = opt
        if cuda:
            self.gtv1.cuda()

    def load(self, p1, p2):
        if self.cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.gtv1.load_state_dict(torch.load(p1, map_location=device))

    def predict(self, sample):
        if self.cuda:
            sample.cuda()
        P = self.gtv1.predict(sample)
        P = self.gtv1.predict(P)

        return P

    def forward(self, sample, debug=False):
        if not debug:
            P = self.gtv1(sample)
            P = self.gtv1(P)
        else:
            P1 = self.gtv1(sample)
            P2 = self.gtv1(P1)
            return P1, P2
        return P


def supporting_matrix(opt):
    dtype = opt.dtype
    cuda = opt.cuda
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
    A = torch.zeros(width ** 2, width ** 2).type(dtype)
    H = torch.zeros(H_dim0, H_dim1).type(dtype)
    for e, p in enumerate(A_pair):
        H[e, p[0]] = 1
        H[e, p[1]] = -1
        A[p[0], p[1]] = 1
        # A[p[1], p[0]] = 1

    opt.I = I  # .type(dtype).requires_grad_(True)
    opt.pairs = A_pair
    opt.H = H  # .type(dtype).requires_grad_(True)
    opt.connectivity_full = A.requires_grad_(True)
    opt.connectivity_idx = torch.where(A > 0)

    for e, p in enumerate(A_pair):
        A[p[1], p[0]] = 1
    opt.logger.info("OPT created on cuda: {0} {1}".format(cuda, dtype))


def _norm(x, newmin, newmax):
    return (x - x.min()) * (newmax - newmin) / (x.max() - x.min() + 1e-8) + newmin


def add_noise(image, sigma):
    from experiment_funcs import get_experiment_noise

    y = np.array(Image.open(imagename)) / 255
    noise_type = "gw"
    noise_var = (sigma / 255) ** 2  # Noise variance 25 std
    seed = 0  # seed for pseudorandom noise realization
    noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, y.shape)
    z = np.atleast_3d(y) + np.atleast_3d(noise)
    z_rang = np.minimum(np.maximum(z, 0), 1)
    noisyimagename = imagepath + "noisy\\" + t + "_g.bmp"
    plt.imsave(noisyimagename, z_rang)


def mkdir(d, remove=True):
    try:
        if not os.path.exists(d):
            os.makedirs(d)
        else:
            if remove:
                shutil.rmtree(d)  # Removes all the subdirectories!
            os.makedirs(d)
    except Exception:
        print(
            "Cannot create ", d,
        )


def patch_splitting(dataset, output_dst, patch_size=36, stride=18):
    """Split each image in the dataset to patch size with size patch_size x patch_size
    dataset: path of full size reference images    
    """
    import matplotlib.pyplot as plt
    output_dst_temp = os.path.join(output_dst, "patches")
    output_dst_noisy = os.path.join(output_dst_temp, "noisy")
    output_dst_ref = os.path.join(output_dst_temp, "ref")
    mkdir(output_dst_temp)
    mkdir(output_dst_noisy)
    mkdir(output_dst_ref)

    dataloader = DataLoader(dataset, batch_size=1)
    total = 0
    patch_size = int(patch_size)
    stride = int(stride)
    for i_batch, s in enumerate(dataloader):
        T1 = (
            s["nimg"]
            .unfold(2, patch_size, stride)
            .unfold(3, patch_size, stride)
            .reshape(1, 3, -1, patch_size, patch_size)
            .squeeze()
        )
        T2 = (
            s["rimg"]
            .unfold(2, patch_size, stride)
            .unfold(3, patch_size, stride)
            .reshape(1, 3, -1, patch_size, patch_size)
            .squeeze()
        )
        print(i_batch, dataset.nimg_name[i_batch], T1.shape)
        img_name = dataset.nimg_name[i_batch].split(".")[0]
        img_ext = dataset.nimg_name[i_batch].split(".")[1]
        for i in range(T1.shape[1]):
            img = T1[:, i, :, :].cpu().detach().numpy().astype(np.uint8)
            img = img.transpose(1, 2, 0)
            plt.imsave(
                os.path.join(
                    output_dst_noisy, "{0}_{1}.{2}".format(img_name, i, img_ext)
                ),
                img,
            )
            total += 1
        for i in range(T2.shape[1]):
            img = T2[:, i, :, :].cpu().detach().numpy().astype(np.uint8)
            img = img.transpose(1, 2, 0)
            plt.imsave(
                os.path.join(
                    output_dst_ref, "{0}_{1}.{2}".format(img_name, i, img_ext)
                ),
                img,
            )
    print("total: ", total)


def cleaning(output_dst):
    """Clean the directory after running"""

    output_dst_temp = os.path.join(output_dst, "patches")
    try:
        shutil.rmtree(output_dst_temp)  # Removes all the subdirectories!
    except Exception:
        print("Cannot clean the temporary image patches")
