Title: A Visual Introduction to FixMatch
Date: 2020-03-18 10:00
Modified: 2020-03-18 10:00
Category: illustration
Slug: illustrated-fixmatch
Summary: Learn how to leverage unlabeled data using FixMatch for semi-supervised learning
Status: draft
Authors: Amit Chaudhary

# Motivation


# Concept

## The FixMatch Framework

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

## Existing SSL Methods
- [Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf): 2013  
- [Regularization with stochastic transformations and perturbations for deep semi-supervised learning](https://arxiv.org/abs/1606.04586): 2016  
- [Mutual exclusivity loss for semi-supervised deep learning](https://arxiv.org/abs/1606.03141): 2016  
- [Temporal ensembling for semisupervised learning](https://arxiv.org/abs/1610.02242): 2017  
- [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780): 2017  
- [Virtual adversarial training: a regularization method for supervised and semi-supervised learning](https://arxiv.org/abs/1704.03976): 2018  
- [Mixmatch: A holistic approach to semi-supervised learning.](https://arxiv.org/abs/1905.02249): 2019  
- [Unsupervised data augmentation for consistency training.](https://arxiv.org/abs/1904.12848): April, 2019  
- [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785): 2020

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

## References
- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
- [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785)
- [Unsupervised data augmentation for consistency training](https://arxiv.org/abs/1904.12848)
- [Mixmatch: A holistic approach to semi-supervised learning](https://arxiv.org/abs/1905.02249)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)
