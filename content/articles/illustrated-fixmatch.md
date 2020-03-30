Title: The Illustrated FixMatch for Semi-Supervised Learning
Date: 2020-03-30 10:00
Modified: 2020-03-30 10:00
Category: illustration
Slug: illustrated-fixmatch-semi-supervised
Summary: Learn how to leverage unlabeled data using FixMatch for semi-supervised learning
Status: draft
Authors: Amit Chaudhary


Deep Learning has shown very promising results in the field of Computer Vision. But when applying it to practical domains such as medical imaging or industries, lack of labeled data is a major challenge . 

In practical settings, labeling data is a time consuming and expensive process. There are a lot of images but only a small portion of them can be labeled due to resource constraints. In such settings, how can we leverage the remaining unlabeled images along with the labeled images to improve performance of our model? The answer is semi-supervised learning.

*FixMatch* is a recent approach by Sohn et al. from Google Brain that improved the state of the art in semi-supervised learning. In this post, we will the understand the concept and working mechanism of FixMatch.

# Intuition behind FixMatch
Say we're doing a cat vs dog classification where we have limited labeled data and a lot of unlabelled images of cats and dogs.
![](/images/fixmatch-labeled-vs-unlabeled.png){.img-center}  

Our usual *supervised learning* approach would be to just train a classifier on labeled images and ignore the unlabelled images.
![](/images/fixmatch-supervised-part.png){.img-center}

We know that a model should be able to handle augmentations of an image to improve generalization. This leads to an interesting idea with unlabeled images. 
>> What if we create augmentation of unlabeled images and make the supervised model predict those images. Since it's the same image, the predicted labels should be same for both.

![](/images/fixmatch-unlabeled-augment-concept.png){.img-center}
Thus, even without the labels, we can use the unlabeled images as a part of our training pipeline. This is the basic idea behind FixMatch and many preceding papers it borrows from.

## The FixMatch Pipeline
With the high level idea clear, let's see how this is actually implemented in practice. The overall pipeline


## How FixMatch works

## Code Implementation
The official implementation from the paper authors is available [here](https://github.com/google-research/fixmatch).

## Citation Info (BibTex)
If you found this blog post useful, please consider citing it as:
```
@misc{chaudhary2020fixmatch,
  title   = {A Visual Introduction to FixMatch},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/03/illustrated-fixmatch}}
}
```

## References
- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
- [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785)
- [Unsupervised data augmentation for consistency training](https://arxiv.org/abs/1904.12848)
- [Mixmatch: A holistic approach to semi-supervised learning](https://arxiv.org/abs/1905.02249)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)

## Existing SSL Methods
pseudolabeling: produce artificial label and train model to predict that, also called self-training  
- Probability of error of some adaptive patternrecognition machines: 1965  
- Iterative reclassification procedure for constructing an asymptotically optimal rule of allocation in discriminant analysis. : 1975  
- [Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf): 2013
- Semi-supervised self-training of object detection models : 2005  
- Self-training with noisy student improves ImageNet classification. : 2019  

Consistency regularization: obtain artificial label using model predicted distribution after randomly modifying input or model function  
- Regularization with stochastic transformations and perturbations for deep semi-supervised learning. : 2016  
- [Temporal ensembling for semisupervised learning](https://arxiv.org/abs/1610.02242): 2017

recent trend of SOTA combine diverse mechanism to produce artificial labels:  
- [Virtual adversarial training: a regularization method for supervised and semi-supervised learning](https://arxiv.org/abs/1704.03976):2018  
- [Mixmatch: A holistic approach to semi-supervised learning](https://arxiv.org/abs/1905.02249) : 2019  
- [Unsupervised data augmentation for consistency training.](https://arxiv.org/abs/1904.12848): April, 2019  
- [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785): 2020