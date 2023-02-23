# NU-Net (experimental)
> To be updated with abstract

The supervised deep learning models which have become dominant for segmentation tasks in
bioimages suffer from the cost of gathering enough labels and the high variability of
images across experiments. We propose NU-net, a foreground segmentation neural network
with self-supervised training, which extends the neural style transfer application.
Unlike typical self-supervised neural networks trained for pretext tasks, which usually
not sufficient to conduct downstream tasks, we aim for NU-Net to be capable enough to
perform the foreground segmentation task only with self-supervision in many cases. The
idea is to generalize what we defined as mask-like style and to transfer it to input
images. In addition, taking advantage of the fact that we could further supervise NU-Net
on top of self-supervision, NU-Net surpassed the baseline, reached the same magnitude of
loss 2 times faster and showed better performance than a model trained only with
supervision. In parallel we are starting to apply NU-Net, first on large
neurodevelopemental images in the context of a project with Institut de la Vision and
second by introducing BigAnnotator, a project to develop an open-source annotation
plug-in for Napari viewer capable of dealing with large bioimages as well as to fully
leverage advantages of NU-Net.


## Installation
```sh
git clone https://github.com/sbinnee/nunet.git
cd nunet
pip install -e .
```


## Usage
The repo provides two scripts: one to test and the other to train a NU-Net.

For common steps to test and train NU-Nets,
1. Go to release page https://github.com/sbinnee/nunet/releases

2. Download an archive called `nunet_models.tar` and unzip it in the root directory of
   the repo. It should unzip `config/` folder and `models_filter_slider/` folder.

To test pre-trained NU-Nets,
- Execute `./scripts/test_nunet.py` following its instruction on the top of
   the script.
> Get help message by invoking the script `python scripts/test_nunet.py --help`.

To train the pre-trained NU-Nets
- Execute `./scripts/./scripts/train_nunet.py` folowing its instruction.

> Note that the provided NU-Net models on the releases page are not reproducible until
> we later release private datasets, namely ones prefixed with `LOB_`. Find more details
> in the paper.
