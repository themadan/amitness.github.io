Title: The Illustrated PIRL: Pretext-Invariant Representation Learning
Date: 2020-03-15 10:00
Modified: 2020-03-15 10:00
Category: illustration
Slug: illustrated-pirl
Summary: Learn how PIRL generates image representations invariant to transformation in self-supervised manner
Status: draft
Authors: Amit Chaudhary

The end of 2019 saw a huge surge in the number of self-supervised learning research papers. On December 2019, *Misra et al.* from Facebook AI Research proposed ["PIRL"](https://arxiv.org/abs/1912.01991) (pronounced as "pearl"), a new method for learning image representations.

In this article, I will explain the rationale behind the paper and how it advances the self-supervised representation learning scene further for vision. We will also see how this compares to the current SOTA approach "[SimCLR](https://amitness.com/2020/03/illustrated-simclr/)" (as of March 11, 2020) which improves shortcomings of PIRL.

# Motivation
A number of interesting [self-supervised learning methods](https://amitness.com/2020/02/illustrated-self-supervised-learning/) have been proposed to learn image representations in recent times. Many of these use the idea of setting up a pretext task exploiting some **geometric transformation** to get labels. This includes [Geometric Rotation Prediction](https://amitness.com/2020/02/illustrated-self-supervised-learning/#6-geometric-transformation-recognition), [Context Prediction](https://amitness.com/2020/02/illustrated-self-supervised-learning/#5-context-prediction), [Jigsaw Puzzle](https://amitness.com/2020/02/illustrated-self-supervised-learning/#4-image-jigsaw-puzzle), [Frame Order Recognition](https://amitness.com/2020/02/illustrated-self-supervised-learning/#1-frame-order-verification), [Auto-Encoding Transformation (AET)](https://arxiv.org/abs/1901.04596) among many others.
![](/images/pirl-geometric-pretext-tasks.png){.img-center}

The pretext task is setup such that representations are learnt for a transformed image to predict some property of transformation. For example, for a rotation prediction task, we randomly rotate the image by say 90 degrees and then ask the network to predict the rotation angle. 
![](/images/pirl-generic-pretext-setup.png){.img-center}
As such, the image representations learnt can overfit to this objective of rotation angle prediction and not generalize well on downstream tasks. The representations will be **covariant** with the transformation. It will only encode essential information to predict rotation angle and could discard useful semantic information.
![](/images/pirl-covariant-representation.png){.img-center}


# The PIRL Concept
PIRL proposes a method to tackle the problem of representations being covariant with transformation. It proposes a problem formulation such that representations generated for both original image and transformed image are similar. There are two goals:  

- Make transformation of image similar to original image  
- Make representations of original and transformed image different from other random images in the dataset  
![](/images/pirl-concept.png){.img-center}

> Intuitively this makes sense because even if an image was rotated, it doesn't change the semantic meaning that this image is still a "cat on a surface"

## PIRL Framework
PIRL defines a generic framework to implement this idea. First, you take a original image I, apply a transformation from some pretext task(e.g. rotation prediction) to get transformed image <tt class="math">I^t</tt>. Then, both the images are passed through ConvNet <tt class="math">\theta</tt> with shared weights to get representations <tt class="math">V_I</tt> and <tt class="math">V_{I^T}</tt>. The representation <tt class="math">V_I</tt> of original image is passed through a projection head f(.) to get representation <tt class="math">f(V_I)</tt>. Similarly, a separate projection head g(.) is used to get representation <tt class="math">g(V_{I^T})</tt> for transformed image. These representations are tuned with a loss function such that representations of <tt class="math">I</tt> and <tt class="math">I^t</tt> are similar, while making it different from other random image representations stored in a memory bank.
![](/images/pirl-general-architecture.png){.img-center}

## Step by Step Example
Let's assume we take a training corpus of only 3 RGB images for simplicity.  
[Corpus of 3 images]
The training dataset of 3 images is denoted by D with all images being RGB.  
<tt class="math">D = \{I_1, I_2, I_3\}</tt>  
where number of images is denoted by <tt class="math">|D| = 3</tt>
and <tt class="math">I_n \in R^{HxWx3}</tt>



**Memory Bank**  
To learn better image representations, it's better to compare current image with large number of negative images. One common approach is to use larger batches and consider all other images in this batch as negative. PIRL, however, proposes to use a memory bank which caches representations of all images and use that during training. This solves problem of not requiring large batch sizes.  
[show this concept]  

For our example, the network is initialized with random weights. Then, a foward pass is done for all images in training data and the representations f(V_I) are stored in memory bank.  
[concept of repre. stored in memory bank]

**Training** 
Now, we take minibatches from the training data. Let's assume we take a batch of size 2 in our case.
[batch size of 2 images]

Now, for each image, the image and it's counterpart transformation is passed through network to get representations. The transformation depends on which pretext task is used. Here, we use the rotation pretext as an example.
[single image representation show projection head]

**Tuning Model**


## Transfer Learning
After the model is trained, then the projection heads <tt class="math">f(.)</tt> and <tt class="math">g(.)</tt> are removed and the ResNet-50 encoder is used for downstream tasks.

## Future Work
The authors state two promising areas for improving PIRL and learn better image representations:  
1. Borrow transformation from other pretext tasks  
2. Combine PIRL with clustering-based approaches  

## Citation Info (BibTex)
If you found this blog post useful, please consider citing it as:
```
@misc{chaudhary2020pirl,
  title   = {The Illustrated PIRL: Pretext-Invariant Representation Learning},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/03/illustrated-pirl}}
}
```

## References
- ["Self-Supervised Learning of Pretext-Invariant Representations"](https://arxiv.org/abs/1912.01991)