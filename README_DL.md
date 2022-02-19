
## DL recognition service basic settings

To use this function, nVIDIA GPU is required.
<br>This software has been tested on Windows10 or Ubuntu18.04 + nVIDIA GeForce RTX 2080ti.
<br>※Please note that it may not work depending on your GPU. For CUDA and other library programs, please use the version compatible with your GPU environment.

1. Copy the LIMTrackService folder downloaded from this repository to **directly under the C drive** (e.g., C:/LIMTrackService) or **the home directory** (e.g., ${HOME}/LIMTrackService).
2. Download and install [CUDA10.0](https://developer.nvidia.com/cuda-10.0-download-archive) & [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive).
3. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   Install it **directly under the C drive** (e.g., C:/Miniconda3) or **the home directory** (e.g., ${HOME}/miniconda3).

```bash
environment variable example (Windows10)
- C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin
- C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/extras/CUPTI/libx64
- C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include
- C:/cudnn-10.0/cuda/bin
- C:/Miniconda3
- C:/Miniconda3/Library/bin
- C:/Miniconda3/Scripts
```
   For the Ubuntu version, add the following to ~/.bashrc file.
```bash
. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate base
```


## DL recognition setting of each algorithm (Windows10)
Set up each algorithm to be manipulated on the command line from the Miniconda prompt. The following is a summary of how to set up each algorithm.



#### StarDist (2D)
1. Download the code from the [official repository](https://github.com/stardist/stardist) and save (overwrite copy) it **directly under the C:/LIMTrackService/stardist/ folder**.

2. Launch the Miniconda prompt and execute the following command.
```bash
> conda create -n stardist python=3.7
> conda activate stardist
> pip install tensorflow-gpu==1.13.1 "stardist[tf1]" packaging zmq psutil pywin32==223 natsort opencv-python-headless 
```

#### Cellpose

1. Download [Version 0.6.5](https://github.com/MouseLand/cellpose/archive/refs/tags/v0.6.5.zip) from the [official repository](https://github.com/MouseLand/cellpose) and save (overwrite copy) it **directly under the C:/LIMTrackService/cellpose/ folder**.


2. Launch the Miniconda prompt and execute the following command.
```bash
> conda create -n cellpose python=3.7
> conda activate cellpose 
> conda install pytorch==1.3 cudatoolkit=10.1 -c pytorch
> pip install natsort opencv-python-headless tifffile tqdm scipy numba zmq scikit-image==0.16.2 pywin32==223 psutil numpy
```

#### YOLACT++
1. Install the Visual Studio 2017 Community edition beforehand.

2. Download the code from the [official repository](https://github.com/dbolya/yolact) and save (overwrite copy) it **directly under the C:/LIMTrackService/yolact/ folder**.

3. Launch the Miniconda prompt and execute the following command.
```bash
> conda create -n yolact python=3.7
> conda activate yolact 
> conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
> pip install cython opencv-python pillow==6.0 pycocotools matplotlib zmq scipy psutil==5.6.7 pywin32
> cd C:/LIMTrackService/yolact/external/DCNv2
> python setup.py build develop
> copy /Y C:/LIMTrackService/yolact/fix/config.py C:/LIMTrackService/yolact/data/config.py
```

4. Download [resnet101_reducedfc.pth](https://drive.google.com/uc?id=1tvqFPd4bJtakOlmn-uIA492g2qurRChj&export=download) and copy it into the **/yolact/weights** folder.

#### Matterport MaskR-CNN

1. Download the code from the [official repository](https://github.com/matterport/Mask_RCNN) and save (overwrite copy) it **directly under the C:/LIMTrackService/matterport/ folder**.

2. Launch the Miniconda prompt and execute the following command.
```bash
> conda create -n matterport python=3.7
> conda activate matterport
> pip install  pywin32==223 opencv-python==3.4.2.17 tensorflow-gpu==1.13.1 keras==2.1.6 imgaug==0.2.6 IPython==6.4.0 h5py==2.8.0 psutil==5.6.7 zmq numpy==1.19.3 matplotlib==3.2.2 scipy==1.4.1 scikit-image==0.16.2
> python C:/LIMTrackService/matterport/fix_model.py
```

3. Download [resnet50_reduce.h5](https://drive.google.com/file/d/1-DaCS-j3rEZnYWdyqnmJM8gONI5LQ5NQ/view?usp=sharing) and copy it into the **/matterport/weights** folder.

#### Detectron2 MaskR-CNN
1. Install the Visual Studio 2017 Community edition beforehand.

2. Download the code from [Forks for Windows10](https://github.com/flkspencer/detectron2) and save (overwrite copy) it **directly under the C:/LIMTrackService/detectron2/ folder**.

3. Launch the Miniconda prompt and execute the following command.
```bash
> conda create -n detectron2 python=3.6
> conda activate detectron2 
> conda install pytorch==1.3 torchvision==0.4.1 cudatoolkit=10.1 -c pytorch
> pip install cython opencv-python fvcore==0.1.1.dev200512 zmq psutil
> cd C:/LIMTrackService/detectron2/cocoapi/PythonAPI
> python setup.py build_ext install
> cd C:/LIMTrackService/detectron2
> "C:/Program Files/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvars64.bat"
(※Please modify as appropriate for your VS2017 installation.)
> python setup.py build develop
```

## DL recognition setting of each algorithm (Ubuntu18.04) 
Set up each algorithm to be manipulated on the command line from the ubuntu terminal. The following is a summary of how to set up each algorithm.



#### StarDist (2D)
1. Download the code from the [official repository](https://github.com/stardist/stardist) and save (overwrite copy) it **directly under the ${HOME}/LIMTrackService/stardist/ folder**.

2. Launch the terminal and execute the following command.
```bash
$ conda create -n stardist python=3.7
$ conda activate stardist
$ pip install tensorflow-gpu==1.13.1 "stardist[tf1]" packaging zmq psutil natsort opencv-python-headless scikit-image
```

#### Cellpose
1. Download [Version 0.6.5](https://github.com/MouseLand/cellpose/archive/refs/tags/v0.6.5.zip) from the [official repository](https://github.com/MouseLand/cellpose) and save (overwrite copy) it **directly under the ${HOME}/LIMTrackService/cellpose/ folder**.


2. Launch the terminal and execute the following command.
```bash
$ conda create -n cellpose python=3.7
$ conda activate cellpose 
$ conda install pytorch==1.3 cudatoolkit=10.1 -c pytorch
$ pip install natsort opencv-python-headless tifffile tqdm scipy numba zmq scikit-image==0.16.2 psutil numpy
```

#### Matterport MaskR-CNN
1. Download the code from the [official repository](https://github.com/matterport/Mask_RCNN) and save (overwrite copy) it **directly under the ${HOME}/LIMTrackService/matterport/ folder**.

2. Launch the terminal and execute the following command.
```bash
$ conda create -n matterport python=3.7
$ conda activate matterport
$ pip install opencv-python==3.4.2.17 tensorflow-gpu==1.13.1 keras==2.1.6 imgaug==0.2.6 IPython==6.4.0 h5py==2.8.0 psutil==5.6.7 zmq matplotlib==3.2.2 scipy==1.4.1 scikit-image==0.16.2

```

3. Download [resnet50_reduce.h5](https://drive.google.com/file/d/1-DaCS-j3rEZnYWdyqnmJM8gONI5LQ5NQ/view?usp=sharing) and copy it into the **/matterport/weights** folder.

