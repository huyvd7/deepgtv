from proxgtv.proxgtv import *
import os
import argparse
import numpy as np
from bm3d import bm3d_rgb, BM3DProfile
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
from PIL import Image
import matplotlib.pyplot as plt


def main(t,sigma):
    # Experiment specifications
    #imagename = 'image_Lena512rgb.png'
    imagepath = 'C:\\Users\\HUYVU\\AppData\\Local\\Packages\\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\\LocalState\\rootfs\\home\\huyvu\\real_540\\'
    imagename = imagepath+ 'ref\\' + t + '_r.bmp'
    # Load noise-free image
    y = np.array(Image.open(imagename)) / 255

    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.
    noise_type = 'gw'
    noise_var = (sigma/255)**2  # Noise variance 25 std
    seed = 0  # seed for pseudorandom noise realization

    # Generate noise with given PSD
    # N.B.: For the sake of simulating a more realistic acquisition scenario,
    # the generated noise is *not* circulant. Therefore there is a slight
    # discrepancy between PSD and the actual PSD computed from infinitely many
    # realizations of this noise with different seeds.

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD
    #z = np.atleast_3d(y) + np.atleast_3d(noise)

    #z_rang = np.minimum(np.maximum(z, 0), 1)
    #noisyimagename=imagepath+ 'noisy\\' + t + '_g.bmp'
    #plt.imsave(noisyimagename, z_rang)
    noisyimagename=imagepath+ 'noisy\\' + t + '_g.bmp'
    z = np.array(Image.open(noisyimagename)) / 255
    d = (z-y).flatten()
    u = np.mean(d)
    std= np.sqrt(((d - u)**2).sum()/d.shape[0])
    std = (17/255)**2
    std = noise_var
    noise, psd, kernel = get_experiment_noise(noise_type, std**2, seed, y.shape)
    # Call BM3D With the default settings.
    y_est = bm3d_rgb(z, psd)

    # To include refiltering:
    # y_est = bm3d_rgb(z, psd, 'refilter');

    # For other settings, use BM3DProfile.
    # profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
    # profile.gamma = 6;  # redefine value of gamma parameter
    # y_est = bm3d_rgb(z, psd, profile);

    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm3d_rgb(z, sqrt(noise_var));

    # If the different channels have varying PSDs, you can supply a MxNx3 PSD or a list of 3 STDs:
    # y_est = bm3d_rgb(z, np.concatenate((psd1, psd2, psd3), 2))
    # y_est = bm3d_rgb(z, [sigma1, sigma2, sigma3])

    psnr = get_psnr(y, y_est)
    print("PSNR:", psnr)

    # PSNR ignoring 16-pixel wide borders (as used in the paper), due to refiltering potentially leaving artifacts
    # on the pixels near the boundary of the image when noise is not circulant
    psnr_cropped = get_cropped_psnr(y, y_est, [16, 16])
    print("PSNR cropped:", psnr_cropped)

    # Ignore values outside range for display (or plt gives an error for multichannel input)
    y_est = np.minimum(np.maximum(y_est, 0), 1)
    z_rang = np.minimum(np.maximum(z, 0), 1)
    plt.imsave('{0}.bmp'.format(t), y_est)
    y_est = np.array(Image.open('t.bmp')) / 255

    psnr = get_psnr(y, y_est)
    print("PSNR 2:", psnr)
    mse = ((y_est - y)**2).mean()*255
    print("MSE:", mse)
    #plt.imsave(imagepath+ 'noisy\\' + t + '_g.bmp', z_rang)

    # TEST CV2 PSNR
    try:
        from skimage.metrics import structural_similarity as compare_ssim
    except Exception:
        from skimage.measure import compare_ssim
    import cv2
    opath = '{0}.bmp'.format(t)
    argref = imagename
    d = cv2.imread(opath)
    tref = cv2.imread(argref)
    (score, diff) = compare_ssim(tref, d, full=True, multichannel=True)
    psnr2 = cv2.PSNR(tref, d)
    print('#######################') 
    print('CV2 PSNR, SSIM: {:.2f}, {:.2f}'.format( psnr2, score))
    print('#######################') 
    print('')
    #plt.title("y, z, y_est")
    #plt.imshow(np.concatenate((y, np.squeeze(z_rang), y_est), axis=1))
    #plt.show()
    return psnr, mse

def _main(imgw=324, sigma=25):
    bm3d_res = {'psnr':list(), 'mse':list()}
    for t in ['1', '3', '5', '7', '9']:
        print("Image: ", t)
        _psnr, _mse = main(t, sigma=18)
        bm3d_res['psnr'].append(_psnr)
        bm3d_res['mse'].append(_mse)
    print("MEAN BM3D PSNR, MSE:", np.mean(bm3d_res['psnr']), np.mean(bm3d_res['mse']))
    for t in ['2', '4', '6', '8', '10']:
        print("Image: ", t)
        _psnr, _mse = main(t, sigma=19.7)
        bm3d_res['psnr'].append(_psnr)
        bm3d_res['mse'].append(_mse)
    print("MEAN BM3D PSNR, MSE:", np.mean(bm3d_res['psnr']), np.mean(bm3d_res['mse']))
    #noisetype='real'
noisetype='gauss'
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-w", "--width", help="Resize image to a square image with given width"
    )
    parser.add_argument(
        "--sigma", default=25
    )

    args = parser.parse_args()
    if args.width:
        imgw = int(args.width)
    else:
        imgw = None

    _main(imgw=imgw, sigma=int(args.sigma))
