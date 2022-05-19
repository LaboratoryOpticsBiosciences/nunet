# NU-Net (experimental)
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
