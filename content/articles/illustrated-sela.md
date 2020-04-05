Title: An Illustrated Guide to Self-Labelling Images
Date: 2020-04-05 14:11
Modified: 2020-04-05 14:11
Category: illustration
Slug: illustrated-self-labelling
Summary: A self-supervised method to auto-generate labels via simultaneous clustering and representation learning
Status: draft
Authors: Amit Chaudhary

In the past one year, a number of methods for self-supervised learning of image representations have been proposed. A recent trend in the methods is using Contrastive Learning ([SimCLR](https://amitness.com/2020/03/illustrated-simclr/), [PIRL](https://amitness.com/2020/03/illustrated-pirl/), [MoCo](https://arxiv.org/abs/1911.05722)) which have given very promising results.

However, as we had seen in our survey on  self-supervised learning, there exist many [other problem formulations](https://amitness.com/2020/02/illustrated-self-supervised-learning/) for self-supervised learning. One promising approach is:
> Combine clustering and representation learning together to learn both features and labels simultaneously.

A paper **SeLa** presented at ICLR 2020 by Asano et al. of the Visual Geometry Group(VGG), University of Oxford improves this approach and achieved state of the art results in various benchmarks.  
![](/images/sela-intro.png){.img-center}
The most interesting side-effect of this method is that we can *auto-generate labels for images in some new domain* and then use those labels to bootstrap our regular workflow of training a supervised model with any architecture. Self-Labelling is very practical for industries and domains with scarce labeled data. Let's understand how it works.

## Solving The Chicken and Egg Problem
At a very high level, the Self-Labelling method works as follows:
 
- Generate the labels and then train a model on these labels
- Generate new labels from the trained model
- Repeat the process

![](/images/sela-chicken-egg-problem.png){.img-center}

> But, how will you generate labels for images in the first place without a trained model? This sounds like the chicken-and-egg problem where if chicken came first, what did it hatch from and if egg came first, who laid the egg?

The solution to the problem is to use a randomly initialized network to bootstrap the first set of image labels. This has been shown to work empirically in the [DeepCluster](https://arxiv.org/abs/1807.05520) paper.  

The authors of DeepCluster used a randomly initialized <span style="color: #51677d;">AlexNet</span> and evaluated it on ImageNet. Since the ImageNet dataset has 1000 classes, if we randomly guessed the classes, we would get an baseline accuracy of <span style="color: #c91212;">1/1000</span> = <span style="color: #c91212;">0.1%</span>. But, a randomly initialized AlexNet was shown to achieve <span style="color: #3fb536;">12%</span> accuracy on ImageNet. This means that a randomly-initialized network possesses some faint signal in its weights.
![](/images/sela-faint-signal.png){.img-center}

Thus, we can use labels obtained from a randomly initialized network to kick start the process which can be refined later.

## Self-Labelling Pipeline
Let's see how this method is implemented in practice with a step by step example of the whole pipeline from the input data to the output labels:  

**1. Training Data**  

First of all, we require N unlabeled images <tt class="math">I_1, ..., I_N</tt> and take batches of them from some dataset such as ImageNet. In the paper, batches of 256 images are used.

**2. Data Augmentation**  
Then random transformation are applied to them. The paper uses the following transformations in sequence:  
    - RandomResizedCrop(224)
    - RandomGrayscale(p=0.2)
    - ColorJitter
    - RandomHorizontalFlip
    - Normalize  
    
Augmentations are applied so that the self-labelling function learned is transformation invariant. 

**3. Choosing Number of Labels/Clusters**  

We then need to choose number of clusters(K) we want to group our data in. By default, ImageNet has 1000 classes so we could use 1000 clusters. This step is dependent on the domain of the data and can be chosen by either domain knowledge or finding optimal number of clusters by looking at model performance. This is denoted by:
<pre class="math">
y_1, ..., y_N \in {1, ..., K}
</pre>
![](/images/sela-best-cluster.png){.img-center}

The paper experimented with the number of clusters ranging from 1000 to 10,000 and found the ImageNet performance improves till 3000 but slightly degrades when using more clusters than that. So the papers uses 3000 clusters and as a result 3000 classes for the output head of the network.

**4. Model Architecture**  
A ConvNet architecture such as AlexNet or ResNet-50 is used as the feature extractor. This network is denoted by <tt class="math">\phi(I)</tt>
and maps an image I to feature vectors <tt class="math">m \in R^D</tt> with dimension D. 

Then, a classification head is used which is simply a single linear layer that converts the feature vectors into class scores. These scores are converted into probabilities using the softmax operator.
<pre class="math">
p(y=.|x_i) = softmax(h\ o\ \phi(x_i) )
</pre>

**5. Initial Random Label Assignment**  
The above model is initialized with random weights and we do a forward pass through the model to get class predictions for each image in the batch. These predicted classes are assumed as the initial labels.

**6. Self Labelling with Optimal Transport**  
Using these generated labels, now we generate the new labels Q by posing this as an instance of optimal transport problem. This problem can be solved efficiently using fast version of the Sinkhorn-Knopp algorithm.

<pre class="math">
E(p, q) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{y=1}^{K} q(y \mid x_i) logp(y \mid x_i)
</pre>

<pre class="math">
min_{p, q} E(p, q)\ subject\ to _y\: q(y\mid x_i) \in \{0, 1\}\ and \sum_{i=1}^{N} q(y | x_i) = N/K
</pre>



Each optimization update involves a single matrix-vector multiplication and so has a complexity of O(NK). So, the method scales linearly with number of images N. The paper shows how they reached convergence within 2 minutes on ImageNet when using GPU to accelerate the optimization. 

**7. Representation Learning**  
Since we have updated labels Q, we can now take predictions of the model on the images and compare it to their corresponding cluster labels with a classification cross-entropy loss. The model is trained for a fixed number of epochs and as the cross-entropy loss decrease, the internal representation learned improves.
<pre class="math">
E(p|y_1, ..., y_N) = -\frac{1}{N} \sum_{i=1}^{N} logp(y_i \mid x_i)
</pre>

**8. Scheduling Cluster Updates**  
The optimization of labels at step 6 is scheduled to occur at most once an epoch. The authors experimented with not using self-labelling algorithm at all to doing the Sinkhorn-Knopp optimization once per epoch. The best result was achieved at 80.
![](/images/sela-optimal-schedule.png){.img-center}

This shows that doing only random-initialization and augmentation is not enough. Self-labeling is giving us a good increase in performance compared to no self-labeling.


```

- Choose a number of clusters default ncl = 3000, head count(hc) = 10, architecture=alexnet
- batch size bs=256, augs=3, paugs=3, epochs=nepochs=400, number of pseudo-operations nopts=100, workers=24,
- Creates model:
    - output classes: outs = [3000]*1 = [3000]
- Training data taken from ImageNet
- augmentations used:
    - Batch size = 256, 24 workers
    - self.L = [N] -> total images
- Optimizer:
    - SGD: weight decay=1e-5, momentum=0.9, learning_rate
    - for batch:
        prediction = model(batch)
        cross-entropy loss against self.L label
        for some unique batch:
            PS = N*K []
            P = softmax(model(data))
            PS[_selected, :] = P
            
            PS -> PS.T
            r = 1/K
            c = 1/N
            solve ->
            update self.L based on argmaxes of the matrix
                # K*N
                argmaxes = np.nanargmax(self.PS, 0) # size N
                self.L[nh] = newL.to(self.dev)
                
            then prediction and compare to that label
```    
            

## Code Implementation
The official implementation of Self-Labelling in PyTorch by the paper authors is available [here](https://github.com/yukimasano/self-label).

## Citation Info (BibTex)
If you found this blog post useful, please consider citing it as:
```
@misc{chaudhary2020SeLa,
  title   = {An Illustrated Guide to Self-Labelling Images},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/03/illustrated-sela}}
}
```

## References
- [Self-labelling via simultaneous clustering and representation learning](https://arxiv.org/abs/1911.05371)
- [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520)