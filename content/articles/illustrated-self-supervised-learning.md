Title: Illustrated Survey of Self-Supervised Learning
Date: 2020-02-22 03:00
Modified: 2020-02-22 03:00
Category: research
Slug: illustrated-self-supervised-learning
Summary: A visual guide to pretext tasks used for self-supervised representation learning
Status: draft
Authors: Amit Chaudhary

# Image  
## 1. **Image Colorization**
Formulation:   
> What if we prepared pairs of (grayscale, colorized) images by applying grayscale to millions of images we have freely available?  

![](/images/ss-colorization-data-gen.png){.img-center}  

We could use a encoder-decoder architecture based on fully convolutional neural network and compute the L2 loss between the predicted and actual color images.

![](/images/ss-image-colorization.png){.img-center}    

To solve this task, the model has to learn about different objects present in image and related parts so that it can paint those parts in the same color. Thus, representations learned are useful for downstream tasks.
![](/images/ss-colorization-learning.png){.img-center}  

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

To solve this pre-text task, we can use architecture similar to that of jigsaw puzzle. We pass the patches through two siamese ConvNets to extract features, concat the features and do a classification over 8 classes denoting the 8 neighbors.
![](/images/ss-context-prediction-architecture.png){.img-center}

**Papers**:  
[Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)

## References
- Jing, et al. “[Self-Supervised Visual Feature Learning with Deep Neural Networks: A Survey.](https://arxiv.org/abs/1902.06162)”