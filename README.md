# Temporal Action Localization by Structured Maximal Sums

This repo contains training and testing code for:

Zehuan Yuan, Jonathan Stroud, Tong Lu, and Jia Deng.
	**Temporal Action Localization by Structured Maximal Sums**.
	CVPR 2017.

## Usage Guide ##

### Prerequisites ###
Our code requires the following dependencies:

- [leveldb-matlab](https://github.com/kyamagu/matlab-leveldb)
- [denseflow](https://github.com/yjxiong/dense_flow)
- cudnn5.0

This repo installs a modified version of [Caffe](http://caffe.berkeleyvision.org/), so it is not a required dependency. However, all caffe dependencies are required.

We recommend a local installation of all dependencies.

GPU(s) are required for optical flow extraction, model training, and testing. 

### Installation & Data Preparation ###

1. Clone this repository:
```
git clone git@github.com:shallowyuan/struct-max-sum.git
```

2. Rename `Makefile.config.example` to `Makefile.config` and edit it following the [caffe installation instructions](http://caffe.berkeleyvision.org/installation.html#compilation).
Note: Our code is only tested on `opencv-2.4.13`.

3. Make training code:
```
cmake .
make all -j8
make matcaffe
```

4. Make testing code:
```
cd action_matlab/tools
make clean
make all
```

5. Additionally install leveldb-matlab in `action_matlab` to make testing work.

Note: our code trains and evaluates on [Thumos14](http://crcv.ucf.edu/THUMOS14/download.html). To train or test using other datasets, follow the instructions in `data` to prepare the database.

All videos are extracted into individual frames, and stored using leveldb. Dense optical flow is extracted using [denseflow](https://github.com/yjxiong/dense_flow).

## Testing ##

1. Download our pretrained models:
```
cd data/models
bash ../../get_reference_models.sh
```

2. Extract confidence scores.
Our model first extracts confidence scores for each video, which are saved into `action_matlab/tools/testmat`.
The input `type` to `model_def` can be used to differentiate between flow and rgb.
```
cd action_matlab/tools
mkdir testmat
matlab 
model_inference(model,model_def,type)
```

3. Get final predictions:
```
matlab < sliding_infer.m
```

4. Evaluate performance.
Detections from `sliding_infer` are stored in `result.txt`. We evaluate performance of these detections using the official [Thumos14 metric](https://storage.googleapis.com/www.thumos.info/thumos15_zips/THUMOS14_evalkit_20150930.zip).
```
matlab
TH14evalDet(<detfilename>,<gtpath>,<subset>,<threshold>)
```

## Training ##
1. Follow the instructions in `data` to prepare the dataset.

2. Download our intialization models:
```
cd data/models
bash ../scripts/get_init_models.sh
```

3. [Optional] reconfigure models.
We have provided the necessary configurations for models in the paper, as well as pre-trained models used for initialization. To try differnet configurations, you can change the prototxt parameters `structsvm_loss_parameter` in `models/action_detection`.

4. Train the RGB model:
```
bash scripts/train.sh
```
Models will be saved in `models`.
Note: Training has only been tested on one GPU. 

## Citation ##

Please cite our paper if you use this repository in your research. BibTeX:
```
@inproceedings{yuan2016temporal,
  title={Temporal Action Localization by Structured Maximal Sums},
  author={Yuan, Zehuan and Stroud, Jonathan and Lu, Tong and Deng, Jia},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

Please direct any questions to Zehuan Yuan `zhyuan001@gmail.com`.

## Acknowledgements ##

A large portion of the code is based on [Caffe](http://caffe.berkeleyvision.org/), made possible by the [Berkeley Vision and Learning Center](http://bair.berkeley.edu/) and many other [contributors](https://github.com/BVLC/caffe/graphs/contributors). 
