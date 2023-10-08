# 3DFSL

The repository contains the official code and dataset for these papers:

[**What Makes for Effective Few-shot Point Cloud Classification?**](https://openaccess.thecvf.com/content/WACV2022/papers/Ye_What_Makes_for_Effective_Few-Shot_Point_Cloud_Classification_WACV_2022_paper.pdf) [WACV 2022]

[**A Closer Look at Few-Shot 3D Point Cloud Classification**](https://link.springer.com/article/10.1007/s11263-022-01731-4) [IJCV 2023]


### Abstract
In recent years, research on few-shot learning (FSL) has been fast-growing in the 2D image domain due to the less requirement 
for labeled training data and greater generalization for novel classes. However, its application in 3D point cloud data 
is relatively under-explored. Not only need to distinguish unseen classes as in the 2D domain, 3D FSL is more challenging 
in terms of irregular structures, subtle inter-class differences, and high intra-class variances when trained on a low number 
of data. Moreover, different architectures and learning algorithms make it difficult to study the effectiveness of existing 
2D FSL algorithms when migrating to the 3D domain. In this work, for the first time, we perform systematic and extensive 
investigations of directly applying recent 2D FSL works to 3D point cloud related backbone networks and thus suggest a 
strong learning baseline for few-shot 3D point cloud classification. Furthermore, we propose a new network, 
Point-cloud Correlation Interaction (PCIA), with three novel plug-and-play components called Salient-Part Fusion (SPF) 
module, Self-Channel Interaction Plus (SCI+) module, and Cross-Instance Fusion Plus (CIF+) module to obtain more 
representative embeddings and improve the feature distinction. These modules can be inserted into most FSL algorithms 
with minor changes and significantly improve the performance. Experimental results on three benchmark datasets, 
ModelNet40-FS, ShapeNet70-FS, and ScanObjectNN-FS, demonstrate that our method achieves state-of-the-art performance 
for the 3D FSL task.  

## Datasets for 3DFSL
To evaluate objectively, we carefully split the existing 3D datasets and construct three
benchmark datasets, [ModelNet40-FS](https://drive.google.com/drive/folders/18WTIClNOWMM9s6mjhwhfLBTNzINZLfRf?usp=drive_link),
 [ShapeNet70-FS](https://drive.google.com/drive/folders/1DiRlJ7dB7YrGuc90_2NcpNL7MK3eQBZh?usp=drive_link) 
 and [ScanObjectNN-FS](https://drive.google.com/drive/folders/1As3Q0-NPDwJn_9xHviftJIQyhrFcHWrh?usp=drive_link), for 3D few-shot point cloud classification
under different scenarios. You can download them from the google drive and put them in `data/` directory. 
We greatly thank the contributions to the original datasets, please cite their works when using our few-shot splits.


## Dependencies
Create a new conda environment:
```
conda create -n 3dfsl python==3.6.x
```
Our experiments conducted on CUDA 10.2 with Pytorch 1.4. You can install pytorch with the following cmds:
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

You also need to install the following packages:
```
h5py
tqdm
matplotlib
tensorboardX
qpth
cvxpy  (conda install -c conda-forge cvxpy)
```

### Installation
Clone this repository:
```
git clone https://github.com/cgye96/A_Closer_Look_At_3DFSL.git
cd A_Closer_Look_At_3DFSL
```
Activate the conda environment:
```
conda activtae 3dfsl
```
Download the pre-processed datasets [ModelNet40-FS](https://drive.google.com/drive/folders/18WTIClNOWMM9s6mjhwhfLBTNzINZLfRf?usp=drive_link),
 [ShapeNet70-FS](https://drive.google.com/drive/folders/1DiRlJ7dB7YrGuc90_2NcpNL7MK3eQBZh?usp=drive_link) 
 and [ScanObjectNN-FS](https://drive.google.com/drive/folders/1As3Q0-NPDwJn_9xHviftJIQyhrFcHWrh?usp=drive_link), put them in ``` ./data```.
    
### Training and Testing
Train and test the model from scratch:
```
python main.py  --mode 'train' \
                --dataset 'ModelNet40_FS' \
                --backbone 'PointNet' \
                --method 'protonet' \
                --exp '_benchmark' \
                --note 'PN' \
                \
                --way 5 \
                --shot 1 \
                --k_fold 5 \
                \
                --stops 30 \
                --step 5
```

## Acknowledgments
Our code builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

Framework:  [**CloserLookFewShot**](https://github.com/wyharveychen/CloserLookFewShot).

Backbone:   [**PointNet**](https://github.com/fxia22/pointnet.pytorch),
 [**PointNet++**](https://github.com/erikwijmans/Pointnet2_PyTorch), 
 [**DGCNN**](https://github.com/WangYueFt/dgcnn), 
 [**PointCNN**](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/points/point_cnn.py),
 [**DensePoint**](https://github.com/Yochengliu/DensePoint), 
 [**RSCNN**](https://github.com/Yochengliu/Relation-Shape-CNN)

2D FSL: [**ProtoNet**](https://github.com/wyharveychen/CloserLookFewShot/blob/master/methods/protonet.py), [**Relation Network**](https://github.com/wyharveychen/CloserLookFewShot/blob/master/methods/relationnet.py), [**MAML**](https://github.com/wyharveychen/CloserLookFewShot/blob/master/methods/maml.py), [**MetaOpt**](https://github.com/kjunelee/MetaOptNet), [**FSLGNN**](https://github.com/vgsatorras/few-shot-gnn)

Thanks very much for their contributions to the community.

### Citation
If you use this code for your research, please cite our papers:
```
@inproceedings{ye2022makes,
  title={What Makes for Effective Few-Shot Point Cloud Classification?},
  author={Ye, Chuangguan and Zhu, Hongyuan and Liao, Yongbin and Zhang, Yanggang and Chen, Tao and Fan, Jiayuan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1829--1838},
  year={2022}
}

@article{ye2022closer,
  title={A Closer Look at Few-Shot 3D Point Cloud Classification},
  author={Ye, Chuangguan and Zhu, Hongyuan and Zhang, Bo and Chen, Tao},
  journal={International Journal of Computer Vision},
  pages={772-795},
  year={2023},
  publisher={Springer}
}
```
