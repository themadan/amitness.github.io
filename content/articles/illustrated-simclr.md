Title: The Illustrated SimCLR Framework
Date: 2020-03-01 10:00
Modified: 2020-03-01 10:00
Category: illustration
Slug: illustrated-simclr
Summary: A visual guide to the SimCLR framework for contrastive learning of visual representations
Status: draft
Authors: Amit Chaudhary

In recent years, [numerous self-supervised learning methods](https://amitness.com/2020/02/illustrated-self-supervised-learning/) have been proposed, improving the performance on various tasks. But, their performance was still below the supervised counterparts. This changed when Chen et. al proposed a new framework in their paper "[SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)". It not only improved upon the previous state-of-the-art self-supervised learning methods but also beat the supervised learning method on ImageNet.

In this article, I will explain the key ideas of the framework proposed in the paper.

## The Nostalgic Intuition
As a kid, I remember we had to solve such puzzles in our textbook.  
![](/images/contrastive-find-a-pair.png)  
The way a child would solve it is look at the picture of animal on left side, know its a cat, then search for cat on the right side.  
![](/images/contrastive-puzzle.gif)  
> "Such exercises were prepared for the child to be able to recognize an object and contrast that to other objects. Can we teach machines in a similar manner?"

It turns out that we can through an idea called contrastive learning. It attempts to teach machines to distinguish between similar and dissimilar things.

## Problem Formulation for Machines
In order to model the above exercise for a machine instead of a child, we can observe that we require 3 things:  

1. **Examples of similar and dissimilar images**   
We would require example pairs of images that are similar and images that are different for training a model.  
![](/images/contrastive-need-one.png){.img-center}  
The supervised school of thought would require us to manually create such pairs by a human. Self-supervised learning can be applied to automate this. But how?
![](/images/contrastive-supervised-approach.png){.img-center}  
![](/images/contrastive-self-supervised-approach.png){.img-center}  

2. **Ability to know what an image represents**  
We need some mechanism to get representations that allows machine to understand an image.
![](/images/image-representation.png){.img-center}

3. **Ability to quantify if two images are similar**  
We need some mechanism to compute similarity of two images. 
![](/images/image-similarity.png){.img-center}

## The SimCLR Framework Approach

The paper proposes a framework "**SimCLR**" for modeling the above problem in a self-supervised manner. It blends the concept of *Contrastive Learning* with a few novel ideas to learn visual representations without human supervision. 

## Framework
The framework is very simple. An image is taken and random transformations are applied to it to get a pair of images. Each image in that pair is passed through an encoder to get representations. Then a non-linear fully connected layer is applied to get representations z. The task is to maximize the similarity between these two representations for same image.
![](/images/simclr-general-architecture.png){.img-center}


## Example
Let's explore the various components of the framework with an example. Suppose we have a training corpus of millions of unlabeled animal images.

1. **Training Data**  
We can use unlabeled images as the training data.

2. **Data Augmentation**  
3. **Base Encoder**  
4. **Projection Head**  
5. **Loss function**  

## Downstream Tasks


## Objective Results
SimCLR outperformed previous supervised and self-supervised methods on ImageNet.  

- On ImageNet, it achieves 76.5% top-1 accuracy which is 7% improvement over previous SOTA self-supervised method and on-par with supervised ResNet50.  
- When trained on 1% of labels, it achieves 85.8% top-5 accuracy outperforming AlexNet with 100x fewer labels

## References
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)