Paper: [Unrolling of Deep Graph Total Variation for Image Denoising](https://arxiv.org/abs/2010.11290)

GitHub: [huyvd7/deepgtv](https://github.com/huyvd7/deepgtv)
<p align="center">
  <img src="legacy/diff_stat.png" class="img-responsive">
<p align="center"><b>Fig.</b> Trained and tested on different noise distributions.</p>
</p>

# Train
Train GTV
```python
python train_gtv.py --batch 64 --lr 1e-4 -m MODEL_NAME.pkl --epoch 50 --train TRAIN_DATASET --width 36
```

Train DGTV
```python
python train_dgtv.py --batch 32 --lr 1e-4 -m MODEL_NAME.pkl --epoch 50 --train TRAIN_DATASET --width 36
```

Params:
- width: dimension of an image in TRAIN_DATASET

# Test
Test GTV
```python
python test_gtv.py -width 720 -m MODEL_NAME.pkl --stride 18 --train_width 36 -multi 500 -p TEST_DATASET
```

Test DGTV
```python
python test_dgtv.py -width 720 -m MODEL_NAME.pkl --stride 18 --train_width 36 --multi 500 -p TEST_DATASET
```

Params:
- width: desired output image size
- train_width: the width that the model was trained on

# TODO
- [ ] Denoise a given single image
