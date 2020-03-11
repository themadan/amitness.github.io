Title: The Illustrated PIRL: Pretext-Invariant Representation Learning
Date: 2020-03-11 10:00
Modified: 2020-03-11 10:00
Category: illustration
Slug: illustrated-pirl
Summary: Learn how PIRL generates self-supervised image representations invariant to image transformations
Status: draft
Authors: Amit Chaudhary

The end of 2019 saw a surge in the number of self-supervised learning research papers. On December 2019, Misra et al. from Facebook AI Research proposed a new method for learning image representations called ["PIRL"](https://arxiv.org/abs/1912.01991) (pronounced as "pearl")

In this article, I will explain the rationale behind the paper and how it advances the self-supervised representation learning scene further for vision. We will also see how this compares to the current SOTA approach "[SimCLR](https://amitness.com/2020/03/illustrated-simclr/)" (as of March 11, 2020) which improves shortcomings of PIRL.

## Motivation
A number of interesting [self-supervised learning methods](https://amitness.com/2020/02/illustrated-self-supervised-learning/) have been proposed to learn image representations in recent times. Many of these use the idea of setting up a pretext task exploiting some geometric transformation to get labels. This includes "**Geometric Rotation Prediction**", "**Context Prediction**", "**Jigsaw Puzzle**", "**Frame Order Recognition**", "**[Auto-Encoding Transformation (AET)](https://arxiv.org/abs/1901.04596)**" among many others.
![](/images/pirl-geometric-pretext-tasks.png){.img-center}

The pretext task is setup such that representations are learnt for a transformed image to predict some property of transformation. For example, for a rotation prediction task, we randomly rotate the image by say 90 degrees and then ask the network to predict the rotation angle. 
![](/images/pirl-generic-pretext-setup.png){.img-center}
As such, the image representations learnt can overfit to this objective of rotation angle prediction and not generalize well on downstream tasks. The representations will be **covariant** with the transformation. It will only encode essential information to predict rotation angle and could discard useful semantic information.
![](/images/pirl-covariant-representation.png){.img-center}


## The PIRL Solution

![](/images/pirl-general-architecture.png){.img-center}

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