Title: A Visual Exploration of DeepCluster
Date: 2020-04-13 14:11
Modified: 2020-04-13 14:11
Category: illustration
Slug: illustrated-deepcluster
Summary: DeepCluster is a self-supervised method to generate labels via K-means clustering and learn representations
Status: draft
Authors: Amit Chaudhary
Cover: /images/deepcluster-pipeline.png


## Preface
- Clustering unsupervised learning method extensively used in computer vision
- Little work to adapt it to E2E training of visual features on large scale datasets
- scalable
- 

## Why needed?
- Previous method: domain dependent, require expert knowledge to design pretext task to get transferable features
- Makes little assumption about inputs
- Domain specific knowledge not needed
- Learn deep representations specific to domains where annotations are scarce
- give examples of previous approaches

## Deep Cluster
Jointly learn parameters of NN and cluster assignments of resulting features
Iteratively group features with standard clustering algo: k-means
Use subsequent assignment as supervision to update weights of the network
iterates between clustering with k-means the features produced by the convnet and updating its weights by predicting the cluster assignments as pseudo-labels in a discriminative loss

## Deep Cluster Pipeline
Let's now understand how the deep cluster pipeline works with an interactive diagram.
![](/images/deepcluster-pipeline.gif){.img-center}

**Synopsis:**  
As seen in the figure above, unlabeled images are taken and <span style="color: #996625; font-weight: bold;">augmentations</span> are applied to them. Then, an <span style="color: #30792c; font-weight: bold;">ConvNet</span> architecture such as <span style="color: #30792c; font-weight: bold;">AlexNet</span> or <span style="color: #30792c; font-weight: bold;">VGG-16</span> is used as the feature extractor. Initially, the <span style="color: #30792c; font-weight: bold;">ConvNet</span> is initialized with randomly weights and we take the <span style="color: #ff787b; font-weight: bold;">feature vector</span> from layer before the final classification head. Then, <span style="color: #34c0c7; font-weight: bold;">PCA</span> is used to reduce the dimension of the <span style="color: #ff787b; font-weight: bold;">feature vector</span> along with whitening and <span style="color: #41adda; font-weight: bold;">L2 normalization</span>. Finally, the processed features are passed to <span style="color: #9559b3; font-weight: bold;">K-means</span> to get cluster assignment for each image.  

These cluster assignments are used as the <span style="color: #9559b3; font-weight: bold;">pseudo-labels</span> and the <span style="color: #30792c; font-weight: bold;">ConvNet</span> is trained to predict these clusters. Cross-entropy loss is used to gauge the performance of the model. The model is trained for 100 epochs with the <span style="color: #9559b3; font-weight: bold;">clustering</span> step occurring once per epoch. Finally, we can take the <span style="color: #ff787b; font-weight: bold;">representations</span> learned and use it for downstream tasks.

## Step by Step Example  

Let's see how DeepCluster is implemented in practice with a step by step example of the whole pipeline from the input data to the output labels:  


**1. Training Data**  
- ImageNet
- 1.3M images uniformly distributed into 1000 classes
- Mini-batches of size = 256

**2. Data Augmentation**  
- Why: Unsupervised methods don’t work directly on color and diff. Strategies have been considered[25, 26]
- Fixed linear transformation based on sobel filters
    - To remove color and increase local contrast
- Central Crop
- Data Augmentation:
    - Random Horizontal Flip
    - Crops of random sizes and aspect ratio
- Why: invariance to data augmentation for feature learning [33]


**3. Choosing Number of Clusters(Labels)**  
10000 clusters -> 10000 classes

**4. Model Architecture**  
- AlexNet
    - 5 conv layers: 96, 256, 384, 384, 256
    - 3 FC layers
    - Remove Local Response Normalization layers
        - Instead use batch normalization
- Alternate: VGG-16 with barch normalization
- Dropout added

**5. The First Epoch** 
- Remove top layer i.e. nn.Linear(4096, classes)
- Get features upto: Linear(4096, 4096, bias=True)
- forward pass with model.eval()
- For each batch:
    - model(batch) -> [256, 4096]
    - features = [23000(len of dataset), 4096]
    - images*features matrix is above

**6. Clustering**  
- Using FAISS: PCA(4096) -> 256
- whitened using eigen_power = -0.5
- L2-normalized: (23000, 256) matrix
- K-means: 
    - Johnson [60] implementation
    - FAISS
    - n_iter=20
- takes 1/3rd of the total trainign time
- cluster update every epoch optimal for ImageNet
- create labels and new dataset by uniformly sampling images from each cluster

**7. Representation Learning** 
- Train like regular model using cluster labels
- batch size = 256
- cross-entropy loss

**8. Training**  
- epochs = 500
- L2 penalization of the weights
- momentum = 0.9, learning rate = 0.05, weight_decay = 10^-5
- Pascal P100 GPU
- constant step size
- NMI calculated why?


## Evaluation and Results
- Freeze all the conv layers
- Add a linear layer and train that
- Trained on ImageNet and YFCC100M
- Outperforms SOTA by significant margin on all standard benchmarks

## Code Implementation of DeepCluster
The official implementation of Deep Cluster in PyTorch by the paper authors is available on [GitHub](https://github.com/facebookresearch/deepcluster). They also provide [pretrained weights](https://github.com/facebookresearch/deepcluster#pre-trained-models) for AlexNet and Resnet-50 architectures.

## Citation Info (BibTex)
If you found this blog post useful, please consider citing it as:
```
@misc{chaudhary2020SeLa,
  title   = {A Visual Exploration of DeepCluster},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/04/illustrated-deepcluster}}
}
```

## References
- [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520)
