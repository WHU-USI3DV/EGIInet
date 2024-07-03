# EGIInet: Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion

This repository contains the official implementation for "Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion" (ECCV 2024) 

**[Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion](https://arxiv.org/abs/2307.08492), ECCV 2024**

## Introduction

This work is accepted by ECCV 2024. In comparison with previous methods that relied on the global semantics of input images, EGIInet efficiently combines the information from two modalities by leveraging the geometric nature of the completion task. Specifically, we propose an explicitly guided information interaction strategy supported by modal alignment for point cloud completion. First, we unified the encoding process to promote modal alignment. Second, we propose a novel explicitly guided information interaction strategy that could help the network identify critical information within images, thus achieving better guidance for completion. Extensive experiments demonstrate the effectiveness of our framework, and we achieved a new state-of-the-art (+16% CD over XMFnet) in benchmark datasets despite using fewer parameters than the previous methods.

## Get Started

### Requirement
- python >= 3.6
- PyTorch >= 1.8.0
- CUDA == 11.8
- easydict
- opencv-python
- transform3d
- h5py
- timm
- open3d
- tensorboardX
- ninja == 1.11.1
- torch-scatter
- einops

Install PointNet++ utils, Density-aware Chamfer Distance and Furthest Point Sampling.
```
cd /models/pointnet2_batch
python setup.py install

cd ../../metrics/CD/chamfer3D/
python setup.py install

cd ../../EMD/
python setup.py install

cd ../../utils/furthestPointSampling/
python setup.py install
```

The code has been trained and tested with Python 3.9, PyTorch 2.1.2 and CUDA 11.8 on Ubuntu 20.04 and RTX 3080Ti.

### Dataset
Download the ShapeNet-ViPC dataset
First, please download the [ShapeNetViPC-Dataset](https://pan.baidu.com/s/1NJKPiOsfRsDfYDU_5MH28A) (143GB, code: **ar8l**). Then run ``cat ShapeNetViPC-Dataset.tar.gz* | tar zx``, you will get ``ShapeNetViPC-Dataset`` contains three floders: ``ShapeNetViPC-Partial``, ``ShapeNetViPC-GT`` and ``ShapeNetViPC-View``. 

For each object, the dataset include partial point cloud (``ShapeNetViPC-Patial``), complete point cloud (``ShapeNetViPC-GT``) and corresponding images (``ShapeNetViPC-View``) from 24 different views. You can find the detail of 24 cameras view in ``/ShapeNetViPC-View/category/object_name/rendering/rendering_metadata.txt``.

Use the code in  ``utils/ViPCdataloader.py`` to load the dataset.

### Pre-trained models
Download the pre-trained models at [Google drive](https://drive.google.com/file/d/1AU2ddmVjbbdEWr5-3w2jqXt9d0nky8ts/view?usp=sharing)

### Evaluation
```
# Specify the checkpoint path in config_vipc.py
__C.CONST.WEIGHTS = "path to your checkpoint"

python main.py --test
```

### Training
```
python main.py
```

## Acknowledgement
The repository is based on [SVDFormer](https://github.com/czvvd/SVDFormer_PointSea), some of the code is borrowed from:
- [Meta-Transformer](https://github.com/invictus717/MetaTransformer)
- [ViPC](https://github.com/Hydrogenion/ViPC)
- [XMFnet](https://github.com/diegovalsesia/XMFnet)

Thanks for their opensourceing.

The point clouds are visualized with blender.

## License

This project is open sourced under MIT license.


