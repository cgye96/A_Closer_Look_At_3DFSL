# 3DFSL

The repository contains the code and dataset for these paper:

[**What Makes for Effective Few-shot Point Cloud Classification?**](https://openaccess.thecvf.com/content/WACV2022/papers/Ye_What_Makes_for_Effective_Few-Shot_Point_Cloud_Classification_WACV_2022_paper.pdf) [WACV 2022]

[**A Closer Look at Few-Shot 3D Point Cloud Classification**](https://link.springer.com/article/10.1007/s11263-022-01731-4) [IJCV 2022]


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

## Dependencies
```
conda create -n 3dfsl python==3.6.x
```
* CUDA 10.2
* Python 3.6
* PyTorch 1.4 (conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch)
* h5py
* tqdm
* plyfile
* matplotlib
* tensorboardX
* cvxpy (conda install -c conda-forge cvxpy)
* qpth

Nota that, the PointNet2, RS-CNN and DensePoint are tested in the CUDA 10.2 with Python 3.6 and PyTorch 1.4. 
If your CUDA version is higher than 10.2, you may not reproduce these backbones.

### Installation
1¡¢Clone this repository:
```
    git clone https://github.com/cgye96/FSL3D.git
    cd FSL3D
```
    
2¡¢We have split and processed the datasets, you can download [ModelNet40_FS], [ShapeNet70_FS] and [ScanObjectNN_FS], and put them in ``` ./dataset```

### Training and Testing
    bash run.sh  
    
    
### 
    

    
## Acknowledgments
Our code builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

Framework:  [**CloserLookFewShot**](https://github.com/wyharveychen/CloserLookFewShot).

Backbone:   [**PointNet**], [**PointNet++**], [**DGCNN**], [**PointCNN**], [**DensePoint**], [**RSCNN**]

2D FSL: [**ProtoNet**], [**Relation Network**], [**MAML**], [**MetaOpt**], [**FSLGNN**], [**Meta-Lstm**]
