Title: The Illustrated Self-Supervised Learning
Date: 2020-02-25 03:00
Modified: 2020-02-25 03:00
Category: illustration
Slug: illustrated-self-supervised-learning
Summary: A visual introduction to the patterns of problem formulation in self-supervised learning
Status: published
Authors: Amit Chaudhary


Yann Lecun, in his [talk](https://www.youtube.com/watch?v=7I0Qt7GALVk&t=2639s), introduced the "cake analogy" to illustrate the importance of self-supervised learning. We have seen this in the Natural Language Processing field where recent developments (Word2Vec, Glove, ELMO, BERT) have embraced self-supervision and achieved state of the art results.
> “If intelligence is a cake, the bulk of the cake is self-supervised learning, the icing on the cake is supervised learning, and the cherry on the cake is reinforcement learning (RL).”  
  
  
Curious to know how self-supervised learning have been applied in the computer vision field, I read up on existing literature on self-supervised learning applied to computer vision through a [recent survey paper](https://arxiv.org/abs/1902.06162) by Jing et. al. 

This post is my attempt to provide an intuitive visual summary of the patterns of problem formulation in self-supervised learning.


# Computer Vision
## 1. **Image Colorization**
Formulation:   
> What if we prepared pairs of (grayscale, colorized) images by applying grayscale to millions of images we have freely available?  

![](/images/ss-colorization-data-gen.png){.img-center}  

We could use a encoder-decoder architecture based on fully convolutional neural network and compute the L2 loss between the predicted and actual color images.

![](/images/ss-image-colorization.png){.img-center}    

To solve this task, the model has to learn about different objects present in image and related parts so that it can paint those parts in the same color. Thus, representations learned are useful for downstream tasks.
![](/images/ss-colorization-learning.png){.img-center}  

**Papers:**  
[Colorful Image Colorization](https://arxiv.org/abs/1603.08511) | [Real-Time User-Guided Image Colorization with Learned Deep Priors](https://arxiv.org/abs/1705.02999) | [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)

## 2. **Image Superresolution**
Formulation:   
> What if we prepared training pairs of (small, upscaled) images by downsampling millions of images we have freely available?  

![](/images/ss-superresolution-training-gen.png){.img-center}  


GAN based models such as [SRGAN](https://arxiv.org/abs/1609.04802) are popular for this task. A generator takes a low-resolution image and outputs a high-resolution image using fully convolutional network. The actual and generated images are compared using both mean-squared-error and content loss to imitate human like quality comparison. A binary-classification discriminator takes an image and classifies whether it's an actual high resolution image(1) or a fake generated superresolution image(0). This interplay between the two models leads to generator learning to produce images wth fine details. 
![](/images/ss-srgan-architecture.png)

Both generator and discriminator learn semantic features which can be used for downstream tasks.

**Papers**:  
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)


## 3. **Image Inpainting**
Formulation:   
> What if we prepared training pairs of (corruped, fixed) images by randomly removing part of images?  

![](/images/ss-image-inpainting-data-gen.png){.img-center}  


Similar to superresolution, we can leverage a GAN-based architecture where the Generator can learn to reconstruct the image while discriminator separates real and generated images.
![](/images/ss-inpainting-architecture.png)

For downstream tasks, [Pathak et al.](https://arxiv.org/abs/1604.07379) have shown that semantic features learnt by such generator gives 10.2% improvement over random initialization on the [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) semantic segmentation challenge, while giving <4% improvements over classification and object detection.

**Papers**:  
[Context encoders: Feature learning by inpainting](https://arxiv.org/abs/1604.07379)

## 4. **Image Jigsaw Puzzle**
Formulation:   
> What if we prepared training pairs of (shuffled, ordered) puzzles by randomly shuffling patches of images?  

![](/images/ss-image-jigsaw-data.png){.img-center}  

Even with only 9 patches, there can be 362880 possible puzzles. To overcome this, only a subset of possible permutations is used such as 64 permutations with highest hamming distance.
![](/images/ss-jigsaw-permutations.png){.img-center}

Suppose we use a permutation that changes the image as shown below. Let's use the permutation number 64 from our total available 64 permutations.
![](/images/ss-jigsaw-permutation-64.png){.img-center}

Now, to recover back the original patches, [Noroozi et al.](https://arxiv.org/abs/1603.09246)
 proposed a neural network called context-free network (CFN) as shown below. Here, the individual patches are passed through the same siamese convolutional layers that have shared weights. Then, the features are combined in a fully-connected layer. In the output, the model has to predict which permutation was used from the 64 possible classes. If we know the permutation, we can solve the puzzle.
![](/images/ss-jigsaw-architecture.png){.img-center}

To solve the Jigsaw puzzle, the model needs to learn to identify how parts are assembled in an object, relative positions of different parts of objects and shape of objects. Thus, the representations are useful for downstream tasks in classification and detection.

**Papers**:  
[Unsupervised learning of visual representions by solving jigsaw puzzles](https://arxiv.org/abs/1603.09246)

## 5. **Context Prediction**
Formulation:   
> What if we prepared training pairs of (image-patch, neighbor) by randomly taking image patch and one of its neighbors around it from large, unlabeled image collection?  

![](/images/ss-context-prediction-gen.png){.img-center}  

To solve this pre-text task, [Doersch et al.](https://arxiv.org/abs/1505.05192) used an architecture similar to that of jigsaw puzzle. We pass the patches through two siamese ConvNets to extract features, concatenate the features and do a classification over 8 classes denoting the 8 possible neighbor positions.
![](/images/ss-context-prediction-architecture.png){.img-center}

**Papers**:  
[Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)

## 6. **Geometric Transformation Recognition**
Formulation:   
> What if we prepared training pairs of (rotated-image, rotation-angle) by randomly rotating images by (0, 90, 180, 270) from large, unlabeled image collection?  


![](/images/ss-geometric-transformation-gen.png){.img-center}  

To solve this pre-text task, [Gidaris et al.](https://arxiv.org/abs/1505.05192) propose an architecture where a rotated image is passed through a ConvNet and the network has to classify it into 4 classes(0/90/270/360 degrees).
![](/images/ss-geometric-transformation-architecture.png){.img-center}

Though a very simple idea, the model has to understand location, types and pose of objects in image to solve this task and as such, the representations learnt are useful for downstream tasks.

**Papers**:  
[Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)

## 7. **Image Clustering**
Formulation:   
> What if we prepared training pairs of (image, cluster-number) by performing clustering on large, unlabeled image collection?  


![](/images/ss-image-clustering-gen.png){.img-center}  

To solve this pre-text task, [Caron et al.](https://arxiv.org/abs/1807.05520) propose an architecture called deep clustering. Here, the images are first clustered and the clusters are used as classes. The task of the ConvNet is to predict the cluster label for an input image.
![](/images/ss-deep-clustering-architecture.png){.img-center}

**Papers**:  
[Deep clustering for unsupervised learning of visual features](https://arxiv.org/abs/1807.05520)

## 8. **Synthetic Imagery**
Formulation:   
> What if we prepared training pairs of (image, properties) by generating synthetic images using game engines and adapting it to real images?  


![](/images/synthetic-imagery-data.png){.img-center}  

To solve this pre-text task, [Ren et al.](https://arxiv.org/pdf/1711.09082.pdf) propose an architecture where weight-shared ConvNets are trained on both synthetic and real images and then a discriminator learns to classify whether ConvNet features fed to it is of a synthetic image or a real image. Due to adversarial nature, the shared representations between real and synthetic images get better.
![](/images/ss-synthetic-image-architecture.png){.img-center}

**Papers**:  
[Cross-Domain Self-supervised Multi-task Feature Learning using Synthetic Imagery](https://arxiv.org/pdf/1711.09082.pdf)

## References
- Jing, et al. “[Self-Supervised Visual Feature Learning with Deep Neural Networks: A Survey.](https://arxiv.org/abs/1902.06162)”