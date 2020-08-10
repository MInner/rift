***Recommendation***

- Our GAN based work for facial attribute editing - [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow).

***New***

- We re-implement CycleGAN by **Tensorflow 2**! The old versions are here: [v1](https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch/tree/v1), [v0](https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch/tree/v0).

<hr style="height:1px" />

<p align="center"> <img src="./pics/horse2zebra.gif" width="100%" /> </p>

<hr style="height:1px" />

# <p align="center"> CycleGAN - Tensorflow 2 </p>

Tensorflow 2 implementation of CycleGAN.

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.*

## Exemplar results

### summer2winter

row 1: summer -> winter -> reconstructed summer, row 2: winter -> summer -> reconstructed winter

<p align="center"> <img src="./pics/summer2winter.jpg" width="100%" /> </p>

### horse2zebra

row 1: horse -> zebra -> reconstructed horse, row 2: zebra -> horse -> reconstructed zebra

<p align="center"> <img src="./pics/horse2zebra.jpg" width="100%" /> </p>

### apple2orange

row 1: apple -> orange -> reconstructed apple, row 2: orange -> apple -> reconstructed orange

<p align="center"> <img src="./pics/apple2orange.jpg" width="100%" /> </p>

# Usage

- Environment

    - Python 3.6

    - TensorFlow 1.15

    - OpenCV, scikit-image, tqdm, oyaml

    - *we recommend [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers), then you can create the AttGAN environment with commands below*

        ```console
        conda create -n AttGAN python=3.6

        source activate AttGAN

        conda install -c anaconda opencv

        conda install -c anaconda scikit-image

        conda install -c anaconda tqdm

        conda install -c conda-forge oyaml

        conda install -c anaconda tensorflow-gpu=1.15
        ```

- Dataset

    - download the summer2winter dataset

        ```console
        sh ./download_dataset.sh summer2winter_yosemite
        ```

    - download the horse2zebra dataset

        ```console
        sh ./download_dataset.sh horse2zebra
        ```

    - see [download_dataset.sh](./download_dataset.sh) for more datasets

- Example of training

    ```console
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset summer2winter_yosemite
    ```

    - tensorboard for loss visualization

        ```console
        tensorboard --logdir ./output/summer2winter_yosemite/summaries --port 6006
        ```

- Example of testing

    ```console
    CUDA_VISIBLE_DEVICES=0 python test.py --experiment_dir ./output/summer2winter_yosemite
    ```
