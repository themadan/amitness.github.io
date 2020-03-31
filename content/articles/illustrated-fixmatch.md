Title: The Illustrated FixMatch for Semi-Supervised Learning
Date: 2020-03-31 10:00
Modified: 2020-03-31 10:00
Category: illustration
Slug: fixmatch-semi-supervised
Summary: Learn how to leverage unlabeled data using FixMatch for semi-supervised learning
Status: draft
Authors: Amit Chaudhary


Deep Learning has shown very promising results in the field of Computer Vision. But when applying it to practical domains such as medical imaging, lack of labeled data is a major challenge. 

In practical settings, labeling data is a time consuming and expensive process. Though, you have a lot of images, only a small portion of them can be labeled due to resource constraints. In such settings, how can we leverage the remaining unlabeled images along with the labeled images to improve performance of our model? The answer is semi-supervised learning.

FixMatch is a recent approach by *Sohn et al.* from Google Brain that improved the state of the art in semi-supervised learning(SSL). It is a simpler combination of previous methods such as UDA and ReMixMatch.  In this post, we will the understand the concept and working mechanism of FixMatch.

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

As seen, we train a supervised model on our labeled images with cross-entropy loss. For each unlabeled image, <span style="color:#97621f">weak augmentation</span> and <span style="color: #3fb536">strong augmentations</span> are applied to get two images. The <span style="color:#97621f;">weakly augmented image</span> is passed to our model and we get prediction over classes. The probability for the most confident class is compared to a <span style="color: #CC0066">threshold</span>. If it is above the <span style="color: #CC0066">threshold</span>, then we take that class as the ground label i.e. <span style="color: #b35ae0;">pseudo-label</span>. Then, the <span style="color: #3fb536">strongly augmented</span> image is passed through our model to get a prediction over classes. This <span style="color: #56a2f3;">probability distribution</span> is compared to ground truth <span style="color: #b35ae0;">pseudo-label</span> using cross-entropy loss.


## Pipeline Components
### 1. Training Data and Augmentation
FixMatch borrows this idea from UDA and ReMixMatch to apply different augmentation i.e weak augmentation on unlabeled image for the pseudo-label generation and strong augmentation on unlabeled image for prediction.

**a. Weak Augmentation**  
For weak augmentation, the paper uses a standard flip-and-shift strategy. It includes two simple augmentations:

- **Random Horizontal Flip**  
![](/images/fixmatch-horizontal-flip-gif){.img-center}
It is applied with a probability of 50%. This is skipped for the SVHN dataset since those images contain digits for which horizontal flip is not relevant. In PyTorch, this can be performed using [transforms](https://pytorch.org/docs/stable/torchvision/transforms.html) as:  

```python
from PIL import Image
import torchvision.transforms as transforms

im = Image.open('dog.png')
weak_im = transforms.RandomHorizontalFlip(p=0.5)(im)
```

- **Random Vertical and Horizontal Translation**  
![](/images/fixmatch-translate.gif){.img-center}
This augmentation is applied up to 12.5%. In PyTorch, this can be implemented using following code where 32 is the size of image needed:
```python
import torchvision.transforms as transforms
from PIL import Image

im = Image.open('dog.png')
resized_im = transforms.Resize(32)(im)
translated = transforms.RandomCrop(size=32, 
                                   padding=int(32*0.125), 
                                   padding_mode='reflect')(resized_im)
```

**b. Strong Augmentation**  
These includes augmentations which outputs heavily distorted versions of the input images. FixMatch applies the CutOut augmentation followed by one of RandAugment or CTAugment.

**1. Cutout**  
![](/images/fixmatch-cutout.gif){.img-center}
This augmentation randomly removes a square part of the image and fills it with gray or black color. PyTorch doesn't have implementation of Cutout but we can reuse its `RandomErasing` transformation to apply CutOut effect.
```python
import torch
import torchvision.transforms as transforms

# Image of 520*520
im = torch.rand(3, 520, 520)

# Fill cutout with gray color
gray_code = 127

# ratio=(1, 1) to set aspect ratio of square
# p=1 means probability is 1, so always apply cutout
# scale=(0.01, 0.01) means we want to get cutout of 1% of image area
# Hence: Cuts out gray square of 52*52
cutout_im = transforms.RandomErasing(p=1, 
                                     ratio=(1, 1), 
                                     scale=(0.01, 0.01), 
                                     value=gray_code)(im)
```

**2. AutoAugment Variants**  
Previous SSL work used *AutoAugment*, which trained a Reinforcement Learning algorithm to find augmentations that leads to best accuracy on some proxy task(e.g. CIFAR-10). This is problematic since we require some labeled dataset to learn the augmentation and also due to resource requirements associated with RL.  

So, FixMatch uses one among two variants of AutoAugment:  
**a. RandAugment**  
The idea of Random Augmentation(RandAugment) is very simple.

- First you have a list of possible augmentation with their range of possible magnitudes.
![](/images/fixmatch-randaug-pool.png){.img-center}
- You select random N augmentations from this list. Here, we are selecting any two from the list.
![](/images/fixmatch-randaug-random-N.png){.img-center}
- Then you select a random magnitude M ranging from 1 to 10. We can select a magnitude of 5. This means a magnitude of 50% in terms of percentage as maximum possible M is 10 and so percentage = 5/10 = 50%.
![](/images/fixmatch-randaug-mag-calculation.png){.img-center}
- Now, the selected augmentations are applied on an image in sequence. Each augmentation has a 50% probability of being applied.
![](/images/fixmatch-randaugment-sequence.png){.img-center}  
- The values of N and M can be found by hyper-parameter optimization on a validation set with grid search. In the paper, they use random magnitude from a pre-defined range at each training step instead of a fixed magnitude.
![](/images/fixmatch-randaugment-grid-search.png){.img-center}

**b. CTAugment**  
CTAugment was a augmentation technique introduced in the ReMixMatch paper and uses ideas from control theory to remove the need for Reinforcement Learning in AutoAugment. Here's how it works: 
 
- We have a set of possible transformations like in RandAugment
- Magnitude values for transformations are divided into bins and each bin is assigned a weight. Initially, all bins have a weight of 1.
- Now two transformations are selected at random with equal chances from this set and their sequence forms a pipeline. This is similar to RandAugment.
- For each transformation, a magnitude bin is selected randomly with a probability according to the normalized bin weights
- Now, a labeled example is augmented with these two transformations and passed to the model to get a prediction
- Based on how close the model predictions was to the actual label, the magnitude bins weights for these transformation are updated.
- Thus, it learns to choose augmentations that the model has a high chance to predict a correct label and thus augmentation that fall within the network tolerance.

> Thus, we see that unlike RandAugment, CTAugment can learn magnitude for each transformation dynamically during training. So, we don't need to optimize it on some supervised proxy task and it has no sensitive hyperparameters to optimize.
Thus, this is very suitable for semi-supervised setting where labeled data is scarce.

### 2. Model Architecture
The paper uses wider and shallower variants of ResNet called [Wide Residual Networks](https://arxiv.org/abs/1605.07146) as the base architecture. The exact variant used is Wide-Resnet-28-2 with a depth of 28 and widening factor of 2. Thus, this model is two times wider than ResNet. The model has 1.5 million parameters. This model can be combined with a linear layer with output neurons equal to number of classes (e.g. 10 for CIFAR-10 and 100 for CIFAR-100).

### 3. Model Training and Loss Function
- We prepare batches of the labeled images of size B and unlabeled images of batch size <tt class="math">\mu B</tt>. Here <tt class="math">\mu</tt> is a hyperparameter that decides the relative size of labeled:unlabeled images in a batch. For example, <tt class="math">\mu=2</tt> means that we use twice the number of unlabeled images compared to labeled images.  
 The paper tried increasing values of <tt class="math">\mu</tt> and found that as we increased the number of unlabeled images, the error rate decreases.
![](/images/fixmatch-effect-of-mu.png){.img-center}
<p class="has-text-centered">Source: Figure 3(a) | FixMatch paper</p>
- For the supervised part of the pipeline which is trained on labeled images, we use the regular cross-entropy loss H() for classification task. The total loss for a batch is defined by <tt class="math">l_s</tt> and is calculated by taking average of cross-entropy losses for each image:
<pre class="math">
l_s = \frac{1}{B} \sum_{b=1}^{B} H( P_b, P_m(y | \alpha(x_b))
</pre>
- For the unlabeled images, first we apply weak augmentation to the unlabeled image and get the probability for the highest predicted class by applying argmax. This is the pseudo-label that will be compared with output of model on strongly augmented image.
<pre class="math">
q_b = p_m(y | \alpha(\mu_b) )
</pre>
<pre class="math">
\vec{q_b} = argmax(q_b)
</pre>
- Now, the same unlabeled image is strongly augmented and it's output is compared to our pseudolabel to compute cross-entropy loss H(). The total unlabeled batch loss is denoted by <tt class="math">l_u</tt> and given by:
<pre class="math">
l_u = \frac{1}{\mu B} \sum_{b=1}^{\mu B} l(max(q_b) >= \tau) H( \vec{q_b}, p_m(y | A(\mu b) )
</pre>
Here <tt class="math">\tau</tt> denotes the threshold above which we take a pseudo-label. This loss is similar to the pseudo-labeling loss. The difference is that we're using weak augmentation for labels and strong augmentation for loss.
- We finally combine these two losses to get total loss that we optimize to improve our model. <tt class="math">\lambda_u</tt> is a fixed scalar hyperparameter that decides how much both the unlabeled image loss contribute relative to the labeled loss.
<pre class="math">
loss = l_s + \lambda_u l_u
</pre>
An interesting result comes from <tt class="math">\lambda_u</tt>. Previous works [43, 21, 3, 2, 31] have shown that increasing weight during course of training is good. But, in FixMatch, this comes for free. Since initially, the model is not confident on labeled data, so its output predictions on unlabeled data will be below threshold. As such, the model will be trained only on labeled data. But as the training progress, the model becomes more confident on labeled data and as such, predictions on unlabeled data will also start to cross threshold. As such, the loss will soon start incorporating predictions on unlabeled images as well. This gives us a free form of curriculum learning. Intuitively, this is how we learn as a child. In early years, we learn easy concepts such as addition of single digit number 1+2, 2+2 before going to 2 digit numbers and then to complex concepts like algebra.



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