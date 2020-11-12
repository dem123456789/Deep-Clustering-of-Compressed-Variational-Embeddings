# Deep Clustering of Compressed Variational Embeddings

This is an implementation of [Deep Clustering of Compressed Variational Embeddings](https://arxiv.org/abs/1910.10341) (VAB)

 - Clustering Accuracy  
![ds-lstm](/asset/cluster_acc.png)
 - PSNR  
 ![diagram](/asset/psnr.png)
 
## Requirements
 - Python 3
 - PyTorch 1.0

## Results
 - MNIST
 
|                                        |  Best Clustering Acuraccy  |
|:--------------------------------------:|:----:|
| K-Means | 55.37 |
| GMM | 42.22 |
| VaDE | 95.30 |
| VAB | 71.69 |
## Acknowledgement
*Suya Wu  
Enmao Diao  
Jie Ding  
Vahid Tarokh*
