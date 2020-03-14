import scipy.sparse as ss
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

    def __init__(self, u_min=1e-3):
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
        
        self.u_min = u_min
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 32, 1 * 1 * 32), nn.Linear(1 * 1 * 32, 1),
            nn.ReLU()
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

class gaussian_noise_(object):
    def __init__(self, stddev, mean):
        self.stddev = stddev
        self.mean = mean

    def __call__(self, sample):
        nimg, rimg = sample["rimg"], sample["rimg"]
        noise = Variable(nimg.data.new(nimg.size()).normal_(self.mean, self.stddev))
        nimg = nimg + noise
        
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
        return {"nimg": torch.from_numpy(nimg).type(dtype), "rimg": torch.from_numpy(rimg).type(dtype)}


    
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
    W = w(((F.unsqueeze(-1).repeat(1, 1, 1, 4) - F.unsqueeze(-1).repeat(1, 1, 1, 4).permute(0, 1, 3, 2))**2).sum(axis=1))

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
    Fs = (opt.H.matmul(F)**2).requires_grad_(True)
    W = gauss(Fs.sum(axis=1)).requires_grad_(True)
    Fs.register_hook(printmax)
    W.register_hook(printmax)
    return W

def weights_init_normal(m):
    """
    Initialize weights of convolutional layers
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class OPT():
    def __init__(self, batch_size=100, width=36, connectivity='8', admm_iter=1, prox_iter=1):
        self.batch_size = batch_size
        self.width = width
        self.edges = 0
        self.nodes = width**2
        self.I = None
        self.pairs = None
        self.H = None
        self.connectivity = connectivity
        self.admm_iter = admm_iter
        self.prox_iter = prox_iter
        
class GTV(nn.Module):
    """
    GLR network
    """

    def __init__(self, width=36, prox_iter=5, u_min=1e-3, u_max = 1, lambda_min=1e-9, lambda_max=1e9, cuda=False, opt=None):
        super(GTV, self).__init__()
        self.cnnf = cnnf()
        
        self.opt=opt
        self.wt = width
        self.width = width
        self.cnnu = cnnu(u_min=u_min)
        self.u_min= u_min
        self.cnny = cnny()
        
        if cuda:
            self.cnnf.cuda()
            self.cnnu.cuda()
            self.cnny.cuda()
            
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.cnnf.apply(weights_init_normal)
        self.cnnu.apply(weights_init_normal)
        
    def forward(self, xf, debug=False): #gtvforward
        E = self.cnnf.forward(xf)
        E.register_hook(printmean)
        self.u = self.cnnu.forward(xf)
        u_max =2.5
        if self.u.max() > u_max:
            masks = (self.u > u_max).type(dtype)
            self.u = self.u - (self.u - u_max)*masks

        masks = (self.u > self.u_min).type(dtype)
        self.u = self.u - (self.u - self.u_min)*masks
        u = self.u.unsqueeze(1).repeat(1, 3, 1)
        u = self.u.median()
        # u=1
        # Y = self.cnny.forward(xf).squeeze(0)

        x = torch.zeros(xf.shape[0], xf.shape[1], opt.width**2, 1).type(dtype).requires_grad_(True)
        z = opt.H.matmul(x).requires_grad_(True)

        ###################
        # E = xf
        # Fs = (opt.H.matmul(E.view(E.shape[0], E.shape[1], opt.width**2, 1))**2)
        # w = Fs.sum(axis=1).abs()
        ###################
        Fs = (opt.H.matmul(E.view(E.shape[0], E.shape[1], opt.width**2, 1))**2)
        w = torch.exp(-(Fs.sum(axis=1)) / (2 * 1 ** 2)).requires_grad_(True)
        if debug:
            print(w[0, :, :].sum().data, 'WEIGHT SUM')
            hist = list()
        w = w.unsqueeze(1).repeat(1, 3, 1, 1)
        
        T=opt.admm_iter
        P=opt.prox_iter
        delta=1
        eta=.1
        lagrange = opt.lagrange.requires_grad_(True)

        y = xf.view(xf.shape[0], xf.shape[1], opt.width**2, 1).requires_grad_(True)
        I = opt.I.requires_grad_(True)
        H = opt.H.requires_grad_(True)
        D = torch.inverse(2*opt.I + delta*(opt.H.T.mm(H))).type(dtype).requires_grad_(True)
        for i in range(T):
            # STEP 1
            xhat = D.matmul(2*y - H.T.matmul(lagrange) + delta*H.T.matmul(z)).requires_grad_(True)
            # STEP 2
            for j in range(P):
                grad = (delta*z - lagrange - delta*H.matmul(xhat)).requires_grad_(True)
                z  = proximal_gradient_descent(x=z, grad=grad, W=w, u=u, eta=eta).requires_grad_(True)
                if debug:
                    l = ( (y-xhat).permute(0, 1, 3, 2).matmul(y-xhat) + (u * w * z.abs()).sum())
                    hist.append(l[0, 0, :, :])
            # STEP 3

            lagrange = (lagrange + delta*(H.matmul(xhat) - z)).requires_grad_(True)

        if debug:
            hist = [h.flatten() for h in hist]
            return hist
        xhat = _norm(xhat, 0, 255)
        return xhat.view(xhat.shape[0], xhat.shape[1], opt.width, opt.width)
    
    def predict(self, xf):
        pass

def supporting_matrix(opt):
    width = opt.width
    
    pixel_indices = [i for i in range(width * width)]
    pixel_indices = np.reshape(pixel_indices, (width, width))
    A = connected_adjacency(pixel_indices, connect=opt.connectivity)
    A_pair = np.asarray(np.where(A.toarray() == 1)).T
    opt.edges = A_pair.shape[0]
    H_dim0 = opt.edges
    H_dim1 = width**2    
#     A_pair = np.unique(np.sort(A_pair, axis=1), axis=0)

    I = torch.eye(width**2, width**2).type(dtype)
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
    opt.connectivity = A.requires_grad_(True)
    opt.connectivity_idx = torch.where(A>0)
    opt.lagrange = lagrange.requires_grad_(True)
    delta = 1
    # opt.D = torch.inverse(2*opt.I + delta*(opt.H.T.mm(H))).type(dtype)

def admm(opt, x, y, z, w, delta=1, u=1, lagrange=0, T=1, P=1):
    xhat =  x
    I = opt.I
    H = opt.H
    D = torch.inverse(2*I + delta*(H.T.mm(H))).type(dtype).requires_grad_(True)
    
    zhat = z
    for i in range(T):
        # STEP 1
        xhat = D.matmul(2*y - H.T.matmul(lagrange) + delta*H.T.matmul(z))
        # STEP 2
        for j in range(P):
            grad = (delta*zhat + lagrange - delta*H.matmul(xhat))
            zhat  = proximal_gradient_descent(x=zhat, grad=grad, W=w, u=u)
        # STEP 3
        lagrange = lagrange + delta*(H.matmul(xhat) - zhat).permute(0, 1, 3, 2).matmul(H.matmul(xhat) - zhat)

    return xhat, zhat, lagrange


def proximal_gradient_descent(x, grad, W, u=1, eta=1, debug=False): 
    v = x - eta* grad    
    v = _norm(v,0,255)
    xhat = prox_gtv(w=W, v=v, u=u, eta=eta, debug=debug)
    return xhat

def prox_gtv(w, v, u, eta=1, debug=False):
    masks1 = (( v.abs() -  (eta*w*u).abs() )>0).type(dtype).requires_grad_(True)
    masks2 = (( v.abs() -  (eta*w*u).abs()) <=0).type(dtype).requires_grad_(True)
    v = v - masks1*eta*w*u*torch.sign(v)
    v = v - masks2*v

    return v

def _norm(x, newmin, newmax):
    return (x - x.min())*(newmax-newmin) / (x.max() - x.min() + 1e-8) + newmin

def printmax(x):
    print(x.max().data)
def printmean(x):
    print(x.mean().data)

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
PATH = os.path.join(DST, "GTV.pkl")
batch_size = 100

dataset = RENOIR_Dataset(
    img_dir=os.path.join('C:\\Users\\HUYVU\\AppData\\Local\\Packages\\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\\LocalState\\rootfs\\home\\huyvu\\dgtv\\test'),
    # transform=transforms.Compose([standardize(normalize=False), ToTensor()]),
    transform=transforms.Compose([standardize(normalize=False), ToTensor(), gaussian_noise_(mean=0, stddev=1)]),
)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True#, pin_memory=True
)

width = 36
opt = OPT(batch_size = batch_size, admm_iter=2, prox_iter=1)
supporting_matrix(opt)
lr = 5e-2
total_epoch = 100
print("Dataset: " , len(dataset))
gtv = GTV(width=36, prox_iter = 1, u_max=10, u_min=.5, lambda_min=.5, lambda_max=1e9, cuda=cuda, opt=opt)
if cuda:
    gtv.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(gtv.parameters(), lr=lr)

hist = list()
tstart = time.time()
for epoch in range(total_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):  # start index at 0
        # get the inputs; data is a list of [inputs, labels]
        labels = data["rimg"].float().type(dtype)
        inputs = data['nimg'].float().type(dtype)

        # inputs = torch.autograd.Variable(inputs, requires_grad=True)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = gtv(inputs, debug=0)
        loss = criterion(outputs, labels)    

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if (i+1)%5 == 0:
    print(
        time.ctime(),
        "[{0}] loss: {1:.3f}, time elapsed: {2:.3f}".format(
            epoch + 1, 255 * running_loss / (i + 1), time.time() - tstart
        ), "mean weight: ", gtv.cnnf.layer1[0].weight.mean().data) 
    gtv(inputs,debug=1)
    hist.append(running_loss / (i + 1))
    if ((epoch + 1) % 5 == 0) or (epoch+1)==total_epoch:
        print("save @ epoch ", epoch + 1)
        torch.save(gtv.state_dict(), PATH)
        histW = gtv(inputs, debug=1)
        histW = [h.cpu().detach().numpy() for h in histW]

        print(histW)

torch.save(gtv.state_dict(), PATH)
print("Total running time: {0:.3f}".format(time.time() - tstart))
