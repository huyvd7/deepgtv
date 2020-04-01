from proxgtv.proxgtv import *
import os

class RENOIR_Dataset2(Dataset):
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
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp"]
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

        nimg_name = os.path.join(self.npath, self.nimg_name[idx])
        nimg = cv2.imread(nimg_name)
        rimg_name = os.path.join(self.rpath, self.rimg_name[idx])
        rimg = cv2.imread(rimg_name)

        
        sample = {'nimg': nimg, 'rimg': rimg, 'nn':self.nimg_name[idx], 'rn':self.rimg_name[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        nimg, rimg, nn, rn = sample['nimg'], sample['rimg'], sample['nn'], sample['rn']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        nimg = nimg.transpose((2, 0, 1))
        rimg = rimg.transpose((2, 0, 1))
        return {'nimg': torch.from_numpy(nimg),
                'rimg': torch.from_numpy(rimg), 'nn':nn, 'rn':rn}

class standardize2(object):
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
        nimg, rimg, nn, rn = sample['nimg'], sample['rimg'], sample['nn'], sample['rn']
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
        return {'nimg': nimg,
                'rimg': rimg, 'nn':nn, 'rn':rn}
dtype=torch.FloatTensor
from torch.autograd import Variable

class gaussian_noise_(object):
    def __init__(self, stddev, mean):
        self.stddev = stddev
        self.mean = mean

    def __call__(self, sample):
        nimg, rimg = sample["rimg"].type(torch.FloatTensor), sample["rimg"]
        noise = self.stddev*Variable(torch.zeros(nimg.shape)).normal_()
        nimg = nimg + noise
        masks = (nimg>255).type(dtype)
        nimg = nimg - (nimg - 255)*masks
        masks = (nimg<0).type(dtype)
        nimg = nimg - (nimg)*masks
        return {"nimg": nimg, "rimg": rimg, 'rn':sample['rn']}

import shutil
import torchvision
def main(imgw=324):
    trainp = 'C:\\Users\\HUYVU\\AppData\\Local\\Packages\\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\\LocalState\\rootfs\\home\\huyvu\\train' 
    testset = ['10', '1', '2', '3', '4', '5', '6', '7','8','9']
    dataset = RENOIR_Dataset2(
        img_dir=os.path.join(trainp),
        transform=transforms.Compose([standardize2(w=imgw), ToTensor2(), gaussian_noise_(mean=0, stddev=25)]),
    )
    
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False#, pin_memory=True
    )
    gaussp = 'C:\\Users\\HUYVU\\AppData\\Local\\Packages\\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\\LocalState\\rootfs\\home\\huyvu\\gauss\\'
    noisyp = 'C:\\Users\\HUYVU\\AppData\\Local\\Packages\\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\\LocalState\\rootfs\\home\\huyvu\\gauss\\noisy\\'
    refp = 'C:\\Users\\HUYVU\\AppData\\Local\\Packages\\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\\LocalState\\rootfs\\home\\huyvu\\gauss\\ref\\'

    shutil.rmtree(gaussp, ignore_errors=True)
    shutil.rmtree(noisyp, ignore_errors=True)
    shutil.rmtree(refp, ignore_errors=True)
    os.makedirs(gaussp)
    os.makedirs(noisyp)
    os.makedirs(refp)

    for i, data in enumerate(dataloader, 0): 
        print(data['rn'])
        inputs = data['nimg'].float().type(dtype).squeeze(0)
        img = inputs.cpu().detach().numpy().astype(np.uint8)
        img = img.transpose(1, 2, 0)
        plt.imsave('{0}{1}_g.bmp'.format(noisyp, testset[i]), img )
        inputs = data['rimg'].float().type(dtype).squeeze(0)
        img = inputs.cpu().detach().numpy().astype(np.uint8)
        img = img.transpose(1, 2, 0)
        plt.imsave('{0}{1}_r.bmp'.format(refp, testset[i]), img )

    dataset = RENOIR_Dataset2(img_dir='..\\gauss\\',
                             transform = transforms.Compose([standardize2(),
                                                ToTensor2()])
                            )
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=1)
    # rm -r gauss_batch
    # mkdir gauss_batch
    # mkdir gauss_batch/noisy
    # mkdir gauss_batch/ref
    
    noisyp = '..\\gauss_batch\\noisy'
    refp =   '..\\gauss_batch\\ref'
    shutil.rmtree(noisyp, ignore_errors=True)
    shutil.rmtree(refp, ignore_errors=True)
    os.makedirs(noisyp)
    os.makedirs(refp)
    stride=18
    for i_batch, s in enumerate(dataloader):
        print(i_batch, s['nimg'].size(),
              s['rimg'].size(), len(s['nimg']), s['nn'], s['rn'])
        T1 = s['nimg'].unfold(2, 36, stride).unfold(3, 36, stride).reshape(1, 3, -1, 36, 36).squeeze()
        T2 = s['rimg'].unfold(2, 36, stride).unfold(3, 36, stride).reshape(1, 3, -1, 36, 36).squeeze()
        nnn = s['nn'][0].split('.')[0]
        rn = s['rn'][0].split('.')[0]
        total = 0
        for i in range(T1.shape[1]):
            img = T1[:, i, :, :].cpu().detach().numpy().astype(np.uint8)
            img = img.transpose(1, 2, 0)
            plt.imsave('{0}\\{1}_{2}.png'.format(noisyp, nnn,i), img )
            total += 1
        for i in range(T2.shape[1]):
            img = T2[:, i, :, :].cpu().detach().numpy().astype(np.uint8)
            img = img.transpose(1, 2, 0)
            plt.imsave('{0}\\{1}_{2}.bmp'.format(refp, rn,i), img )
            
        print(total)
        print(noisyp, refp)
    print(T1.shape)

if __name__=="__main__":
    main(imgw=1080)