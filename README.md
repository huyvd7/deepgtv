Paper: [Unrolling of Deep Graph Total Variation for Image Denoising](https://arxiv.org/abs/2010.11290)

GitHub: [huyvd7/deepgtv](https://github.com/huyvd7/deepgtv)
<p align="center">
  <img src="legacy/diff_stat.png" class="img-responsive">
<p align="center"><b>Fig.</b> Trained and tested on different noise distributions.</p>
</p>

# Train DGTV
```python
python train_dgtv.py --batch 32 --lr 1e-4 -m MODEL_NAME.pkl --epoch 50 --train TRAIN_DATASET --width 36
```

Params:
- width: split images in the dataset to patches of size `width x width`

# Test DGTV
```python
python test_dgtv.py -width 720 -m MODEL_NAME.pkl --stride 18 --multi 500 -p TEST_DATASET
```

Params:
- width: desired output image size
- multi: # of patches to be processed simultaneously

# TODO
- [ ] Denoise a given single image
