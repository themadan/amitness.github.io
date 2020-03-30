Title: The Illustrated FixMatch for Semi-Supervised Learning
Date: 2020-03-30 10:00
Modified: 2020-03-30 10:00
Category: illustration
Slug: fixmatch-semi-supervised
Summary: Learn how to leverage unlabeled data using FixMatch for semi-supervised learning
Status: draft
Authors: Amit Chaudhary


Deep Learning has shown very promising results in the field of Computer Vision. But when applying it to practical domains such as medical imaging, lack of labeled data is a major challenge. 

In practical settings, labeling data is a time consuming and expensive process. Though, you have a lot of images, only a small portion of them can be labeled due to resource constraints. In such settings, how can we leverage the remaining unlabeled images along with the labeled images to improve performance of our model? The answer is semi-supervised learning.

FixMatch is a recent approach by *Sohn et al.* from Google Brain that improved the state of the art in semi-supervised learning. In this post, we will the understand the concept and working mechanism of FixMatch.

## Intuition behind FixMatch
Say we're doing a cat vs dog classification where we have limited labeled data and a lot of unlabelled images of cats and dogs.
![](/images/fixmatch-labeled-vs-unlabeled.png){.img-center}  

Our usual *supervised learning* approach would be to just train a classifier on labeled images and ignore the unlabelled images.
![](/images/fixmatch-supervised-part.png){.img-center}

Instead of ignoring unlabeled images, we could instead apply below approach. We know that a model should be able to handle perturbations of an image as well to improve generalization.  
>> What if we create augmented versions of unlabeled images and make the supervised model predict those images. Since it's the same image, the predicted labels should be same for both.

![](/images/fixmatch-unlabeled-augment-concept.png){.img-center}
Thus, even without knowing their correct labels, we can use the unlabeled images as a part of our training pipeline. This is the core idea behind FixMatch and many preceding papers it builds upon.

## The FixMatch Pipeline
With the intuition clear, let's see how FixMatch is actually applied in practice. The overall pipeline is summarized by the following figure:
![](/images/fixmatch-pipeline.png){.img-center}

**Synopsis:**  

As seen, we train a supervised model on our labeled images with cross-entropy loss. For each unlabeled image, <span style="color:#97621f">weak augmentation</span> and <span style="color: #3fb536">strong augmentations</span> are applied to get two images. The <span style="color:#97621f;">weakly augmented image</span> is passed to our model and we get prediction over classes. The probability for the most confident class is compared to a <span style="color: #CC0066">threshold</span>. If it is above the <span style="color: #CC0066">threshold</span>, then we take that class as the ground label i.e. pseudo-label. Then, the <span style="color: #3fb536">strongly augmented</span> image is passed through our model to get a prediction over classes. This <span style="color: #56a2f3;">probability distribution</span> is compared to ground truth <span style="color: #b35ae0;">pseudo-label</span> using cross-entropy loss.


## Pipeline Components

## Code Implementation
The official implementation from the paper authors is available [here](https://github.com/google-research/fixmatch). Unofficial implementations using RandAugment and evaluated on CIFAR-10 and CIFAR-100 in PyTorch are available here ([first](https://github.com/kekmodel/FixMatch-pytorch), [second](https://github.com/CoinCheung/fixmatch), [third](https://github.com/valencebond/FixMatch_pytorch)).

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