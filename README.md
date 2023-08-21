# NU-Net (experimental)
> To be updated with abstract

While supervised deep neural networks have become the dominant method for image analysis
tasks in bioimages, truly versatile methods are not available yet because of the
diversity of modalities and conditions and the cost of retraining. In practice,
day-to-day biological image analysis still largely relies on ad hoc workflows often
using classical linear filters. We propose NU-Net, a convolutional neural network filter
selectively enhancing cells and nuclei, as a drop-in replacement of chains of classical
linear filters in bioimage analysis pipelines. Using a style transfer architecture, a
novel perceptual loss implicitly learns a soft separation of background and foreground.
We used self-supervised training using 25 datasets covering diverse modalities of
nuclear and cellular images. We show its ability to selectively improve contrast, remove
background and enhance objects across a wide range of datasets and workflow while
keeping image content. The pre-trained models are light and practical and available as a
free and open-source software for the community, as well as a ready to use plugin for
Napari.


## Installation
```sh
git clone https://github.com/LaboratoryOpticsBiosciences/nunet.git
cd nunet
pip install -e .
```


## Usage
The repo provides two scripts: one to test and the other to train a NU-Net.

For common steps to test and train NU-Nets,
1. Go to release page https://github.com/LaboratoryOpticsBiosciences/nunet.git

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
