# Multi-view 3D Human Pose Estimation using MMPose

Multi-view 3D Human Pose Estimation on our own dataset for production using [MMPose](https://github.com/open-mmlab/mmpose).

## Installation

```bash
conda create -n mmpose python=3.8 pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y
conda activate mmpose
pip install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
```

## Prepare Datasets

Campus and Shelf Datasets can be downloaded [here](https://campar.in.tum.de/Chair/MultiHumanPose).

### Campus (~1GB)

```bash
wget https://www.campar.in.tum.de/public_datasets/2014_cvpr_belagiannis/CampusSeq1.tar.bz2
mkdir data
tar -jxvf CampusSeq1.tar.bz2 -C data
cd data
mv CampusSeq1 Campus
wget https://github.com/microsoft/voxelpose-pytorch/raw/main/data/panoptic_training_pose.pkl
cd Campus
wget https://github.com/microsoft/voxelpose-pytorch/raw/main/data/CampusSeq1/calibration_campus.json
wget https://github.com/microsoft/voxelpose-pytorch/raw/main/data/CampusSeq1/pred_campus_maskrcnn_hrnet_coco.pkl
```

### Shelf (~16GB)

```bash
wget https://www.campar.in.tum.de/public_datasets/2014_cvpr_belagiannis/Shelf.tar.bz2
mkdir data
tar -jxvf Shelf.tar.bz2 -C data
cd data
wget https://github.com/microsoft/voxelpose-pytorch/raw/main/data/panoptic_training_pose.pkl
cd Shelf
wget https://github.com/microsoft/voxelpose-pytorch/raw/main/data/Shelf/calibration_shelf.json
wget https://github.com/microsoft/voxelpose-pytorch/raw/main/data/Shelf/pred_shelf_maskrcnn_hrnet_coco.pkl
```

## Inference with Pre-trained Models

### Test a dataset

### Run demos

```bash
python demo/body3d_multiview_detect_and_regress_img_demo.py \
    ${MMPOSE_CONFIG_FILE} \
    ${MMPOSE_CHECKPOINT_FILE} \
    --out-img-root ${OUT_IMG_ROOT} \
    --camera-param-file ${CAMERA_FILE} \
    [--img-root ${IMG_ROOT}] \
    [--device ${GPU_ID or CPU}] \
    [--out-img-root ${OUTPUT_DIR}] \
    [--visualize-single-view ${VIS_SINGLE_IMG}]
```

Tips: The parameters in `[]` are optional, and other parameters are required.

- CMU Panoptic

```bash
wget https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5-545c150e_20211103.pth
mkdir demo/data
wget -P demo/data https://download.openmmlab.com/mmpose/demo/panoptic_body3d_demo.tar
tar -xf demo/data/panoptic_body3d_demo.tar -C demo/data
python demo/body3d_multiview_detect_and_regress_img_demo.py \
    configs/body/3d_kpt_mview_rgb_img/voxelpose/panoptic/voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5.py \
    voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5-545c150e_20211103.pth \
    --img-root demo/data/panoptic_body3d_demo \
    --camera-param-file demo/data/panoptic_body3d_demo/camera_parameters.json \
    --visualize-single-view \
    --device cuda:0 \
    --out-img-root vis/panoptic
```

- Ours

```bash
python demo/body3d_multiview_detect_and_regress_img_demo.py \
    configs/body/3d_kpt_mview_rgb_img/voxelpose/panoptic/voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5.py \
    voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5-545c150e_20211103.pth \
    --img-root data/Ours \
    --camera-param-file data/Ours/camera_parameters.json \
    --out-img-root vis/ours \
    --visualize-single-view
```

## Train a model

### Training setting

### Train with a single GPU

### Train with CPU

### Train with multiple GPUs

### Train with multiple machines

### Launch multiple jobs on a single machine
  
## Benchmark

## Issues

- [Pretrained voxelpose with custom data](https://github.com/open-mmlab/mmpose/issues/1310)

> As far as I know, current multiview 3d pose estimation methods like VoxelPose cannot perform well in a new scene, sometimes may fail in the same scene with a different camera setting. So I recommend you to train a your model from scratch. Given that GT 3d poses cannot be easily obtained, you can follow the setting on campus/shelf, where you can artificially generate the training data.