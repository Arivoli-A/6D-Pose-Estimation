# 6D Pose Estimation from RGB Images
This project is a reconstruction of the original PoET architecture (https://github.com/aau-cns/poet), modified to incorporate an asynchronous object detection backbone and a transformer-based encoder-decoder architecture. The pose estimation pipeline runs on a single NVIDIA A5000 GPU.

## Setting up environment and run the model

To set up the conda environment, run the following command:

```bash
. create_env.sh
```
After activating the environment to set the shared libraries in the path, run the following command:

```bash
. lib_path_set.sh
```
Finally to copy the MSDeformAttn class from the build folder, run the following command:

```bash
. deform_attn_file.sh
```
To download the pre computed backbone data, run the following command:

```bash
. download_data.sh
```
Record the path of the folder containing the data, which should be provided as the ```dataset_path``` in ```main.py```.

Then run ```main.py``` file using the following command

```bash
python main.py
```
If we want to use pretrained weights for initial start point, then run the following command:

(NOTE: Pretrained weights are from PoET model (https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/poet-pose-estimation-transformer-for-single-view-multi-object-6d-pose-estimation/#tab-id-2) with YOLO-v4 backbone, so caution regarding strides and shapes of weight is needed before use)

```bash
. download_pre_train_weight.sh
```
