
## Training dataset description

The training dataset should contain pairs of ground-truth and noisy image patches. In the original paper, our dataset contains 10 `720 x 720` images, 
each image is split into multiple patches of size `36 x 36` with `18-pixel` overlapping. Assuming you already split your images into patches with the described specification, if
your train set will be located in TRAINSET_PATH, then
- The noisy patches should be placed in TRAINSET_PATH/noisy
- The ground-truth patches should be placed in TRAINSET_PATH/ref

Ground-truth patch and noisy patch should have the same filename, e.g., TRAINSET_PATH/noisy/0.jpg will be paired with TRAINSET_PATH/ref/0.jpg

## Testing dataset description

The testing dataset should contain pairs of ground-truth and noisy images. There is no need to split patches for the testing data, just use whole images. If your test set will be located in TESTSET_PATH, then
- The noisy patches should be placed in TESTSET_PATH/noisy
- The ground-truth patches should be placed in TESTSET_PATH/ref

Ground-truth patch and noisy patch should have the same filename, e.g., TESTSET_PATH/noisy/1.jpg will be paired with TESTSET_PATH/ref/1.jpg
