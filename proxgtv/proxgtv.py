import scipy.sparse as ss
import torch
import numpy as np
import os
import cv2
import torch.nn as nn
from torch.utils.data import Dataset

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
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )
        self.relu = nn.ReLU()

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

    def __init__(self):
        super(cnnu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 32, 1 * 1 * 32), nn.Linear(1 * 1 * 32, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out + 1e-3


class cnny(nn.Module):
    """
    CNN Y of GLR
    """

    def __init__(self):
        super(cnny, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out = identity + out
        del identity
        return out

class cnnp(nn.Module):
    """
    CNN Y of GLR
    """

    def __init__(self, wt=36):
        super(cnnp, self).__init__()
        self.wt = wt
        self.l1 = nn.Sequential(
            torch.nn.Linear(self.wt**2, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 1),
            torch.nn.ReLU())

    def forward(self, x):
        _lambda = self.l1(x)
        return _lambda
    
class RENOIR_Dataset(Dataset):
    """
    Dataset loader
    """

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.npath = os.path.join(img_dir, "noisy")
        self.rpath = os.path.join(img_dir, "ref")
        self.nimg_name = sorted(os.listdir(self.npath))
        self.rimg_name = sorted(os.listdir(self.rpath))
        self.nimg_name = [
            i
            for i in self.nimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp"]
        ]
        self.rimg_name = [
            i
            for i in self.rimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp"]
        ]
        self.transform = transform

    def __len__(self):
        return len(self.nimg_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        nimg_name = os.path.join(self.npath, self.nimg_name[idx])
        nimg = cv2.imread(nimg_name)
        rimg_name = os.path.join(self.rpath, self.rimg_name[idx])
        rimg = cv2.imread(rimg_name)

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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Swap color axis from H x W x C (numpy) to C x H x W (torch)
        """
        nimg, rimg = sample["nimg"], sample["rimg"]
        nimg = nimg.transpose((2, 0, 1))
        rimg = rimg.transpose((2, 0, 1))
        return {"nimg": torch.from_numpy(nimg), "rimg": torch.from_numpy(rimg)}


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
    fi, fj = F[:, :, ij[0]], F[:, :, ij[1]]
    d = dist(fi, fj)
    return w(d).type(dtype)


def w(d, epsilon=1):
    """
    Compute (3)
    """
    return torch.exp(-d / (2 * epsilon ** 2))


def dist(fi, fj):
    """
    Compute the distance using equation (4)
    """
    return torch.sum((fi - fj) ** 2, axis=1).type(dtype)



def laplacian_construction(opt, F):
    """
    Construct Laplacian matrix
    """
    Fs = F.unsqueeze(-1).repeat(1, 1, 1, F.shape[-1]) 
    W = w(((Fs - Fs.permute(0, 1, 3, 2))**2).sum(axis=1))
    return A_pair

def weights_init_normal(m):
    """
    Initialize weights of convolutional layers
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class GTV(nn.Module):
    """
    GLR network
    """

    def __init__(self, width=36, prox_iter=5, u_max = 1, cuda=False):
        super(GTV, self).__init__()
        self.u = 1
        self.u_max = u_max
        self.cnnf = cnnf()
        self.cnny = cnny()
        self.cnnu = cnnu()
#         self.cnnp = list()
        self.cnnp= nn.ModuleList()
        self.prox_iter = prox_iter
        for i in range(prox_iter):
            self.cnnp.append( cnnp())
        
        self.wt = width
        self.lambda_list = list()
        
        if cuda:
            self.cnnf.cuda()
            
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.cnnf.apply(weights_init_normal)
        for i in range(prox_iter):
            self.cnnp[i].apply(weights_init_normal)
        
        self._lambda = 0
        
    def forward(self, xf):
        
        img_dim = self.wt
        lambda_list = []
        upper_lambda = 1e5
        lower_lambda = 1e-5
        u_max = self.u_max
        E = self.cnnf.forward(xf).squeeze(0)
        Y = self.cnny.forward(xf).squeeze(0)
        self.u = self.cnnu.forward(xf)
        if self.u.max() > u_max:
            masks = (self.u > u_max).type(dtype)
            self.u = self.u - (self.u - u_max)*masks
            
        u = self.u.unsqueeze(1).repeat(1, 3, 1)
        
        A = laplacian_construction(
            width=img_dim, F=E.view(E.shape[0], E.shape[1], img_dim ** 2)
        )
        
        x = Y.view(Y.shape[0],Y.shape[1], img_dim**2)
#         x = xf.view(xf.shape[0], xf.shape[1], img_dim**2)
        W = A.view(A.shape[0], A.shape[1], img_dim**2)
        W = W.sum(axis=2).unsqueeze(1)
        W = W.repeat(1, 3, 1)
        
        # ACCELERATED PROXIMAL GRADIENT
        t = torch.ones(xf.shape[0], xf.shape[1], 1)
        v = x
        out = xf.view(xf.shape[0], xf.shape[1], img_dim**2)
        for i in range(self.prox_iter):
            self._lambda = self.cnnp[i].forward(out)
            masks = (self._lambda < lower_lambda).type(dtype)
            self._lambda = self._lambda - (self._lambda - lower_lambda)*masks
            masks = (self._lambda > upper_lambda).type(dtype)
            self._lambda = self._lambda - (self._lambda - upper_lambda)*masks
            lambda_list.append(self._lambda)

            out = proximal_gradient_descent(x=v, y=x, w=W, u=u, eta=self._lambda)
            
            if i != (self.prox_iter-1):
                old_t = t.clone()
                t = (1 + torch.sqrt(1 + 4*(t**2)))/2
                v = v +( (old_t-1)/t )* (out-v)
        
        self.lambda_list = lambda_list
        return out.view(xf.shape[0], 3, img_dim, img_dim)

    def predict(self, xf):
        pass
    
class DeepGTV(nn.Module):
    """
    Stack 4 GLRs
    """

    def __init__(self, gtv_no = 4, starting_point = None, width=36, cuda=False):
        super(DeepGTV, self).__init__()
        self.gtv_no = gtv_no
        self.gtv = nn.ModuleList()
        if starting_point:
            for i in range(gtv_no):
                gtv = GTV(width=36, prox_iter = 3, cuda=cuda)
                gtv.load_state_dict(starting_point.state_dict())
                self.gtv.append(gtv)
        else:
            for i in range(gtv_no):
                self.gtv.append(GTV(cuda=cuda))
                
        self.cuda = cuda

        if self.cuda:
            for i in range(gtv_no):
                self.gtv[i].cuda()

    def load(self, PATH):
        if self.cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        for i in range(self.gtv_no):
            self.gtv[i].load_state_dict(torch.load(PATH[i], map_location=device))

    def predict(self, sample):
        if self.cuda:
            sample.cuda()
        P = sample
        for i in range(self.gtv_no):
            P = self.gtv[i].predict(P)
        
        return P

    def forward(self, sample):
        P = sample
        for i in range(self.gtv_no):
            P = self.gtv[i].forward(P)
        
        return P
    
def proximal_gradient_descent(x, y, w, u=1, eta=1): 
    grad = eta * (2*x - 2*y)
    x = x - grad
    xhat = prox_gtv(w, x, u, eta)

    return xhat

def prox_gtv(w, v, u, eta=1):
    
    masks = (v.abs() - (eta*w*u).abs() >0).type(dtype)
    v = v - masks*eta*w*u*torch.sign(v)
    masks = (v.abs() - (eta*w*u).abs() <=0).type(dtype)
    v = v - masks*v
    
#     elif w<0:
#         if abs(v) >= abs(eta*w):
#             return 0
#         else:
#             return v - eta * w * np.sign(v)
    return v
