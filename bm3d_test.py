import bm3d
import cv2 
def bm3d_denoise(inp, argref, width=None):

    sample = cv2.imread(inp)
    if width==None:
        width = sample.shape[0]
    else:
        sample = cv2.resize(sample, (width, width))
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample = sample.transpose((2, 0, 1))

    ref = cv2.imread(argref)
    if ref.shape[0] != width or ref.shape[1] != width:
        ref = cv2.resize(ref, (width, width))
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    denoised_image = bm3d.bm3d(sample, sigma_psd=25, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    psnr2 = cv2.PSNR(ref, denoised_image)
    print(psnr2)

