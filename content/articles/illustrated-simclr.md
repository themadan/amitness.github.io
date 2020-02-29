Title: The Illustrated SimCLR Framework
Date: 2020-02-29 10:00
Modified: 2020-02-29 10:00
Category: illustration
Slug: illustrated-simclr
Summary: A visual guide to the SimCLR framework for contrastive learning of visual representations
Status: draft
Authors: Amit Chaudhary

In recent years, [numerous self-supervised learning methods](https://amitness.com/2020/02/illustrated-self-supervised-learning/) have been proposed, improving the performance on various tasks. But, their performance was still below the supervised counterparts. On February 14, 2020, XYZ proposed a new framework in their paper "[SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)". It not only improved upon the previous state-of-the-art self-supervised learning methods but even beat the supervised learning method on ImageNet.

In this article, I will explain the key ideas of the framework proposed in the paper.

## The Nostalgic Intuition
As a kid, I remember we had to solve such puzzles in our textbook.  
![](/images/contrastive-find-a-pair.png)  
The way a child would solve it is look at the picture of animal on left side, know its a cat, then search for cat on the right side.  
![](/images/contrastive-puzzle.gif)  
> Such exercises were prepared for the child to be able to recognize an object and contrast that to other objects. Can we teach machines in a similar manner?

## Contrastive Learning
Contrastive learning formulates the above idea in the context of representation learning. We want to teach a machine to be able to distinguish between similar and dissimilar things.
In order to model the above exercise for a machine instead of a child, we require 3 things:  

1. **Example pairs of similar and dissimilar images**
2. **Ability to know what an image represents**
3. **Ability to quantify if two images are similar**

## References
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)