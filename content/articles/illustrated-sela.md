Title: An Illustrated Guide to Self-Labelling Images
Date: 2020-04-05 14:11
Modified: 2020-04-05 14:11
Category: illustration
Slug: illustrated-sela
Summary: A self-supervised method to auto-generate labels via simultaneous clustering and representation learning
Status: draft
Authors: Amit Chaudhary

In the past one year, a number of methods for self-supervised learning of image representations have been proposed. A recent trend in the methods is using Contrastive Learning ([SimCLR](https://amitness.com/2020/03/illustrated-simclr/), [PIRL](https://amitness.com/2020/03/illustrated-pirl/), [MoCo](https://arxiv.org/abs/1911.05722)) which have given very promising results.

However, as we had seen in our survey on  self-supervised learning, there exist many [other problem formulations](https://amitness.com/2020/02/illustrated-self-supervised-learning/) for self-supervised learning. One promising approach is:
> Combine clustering and representation learning together to learn both features and labels simultaneously.

A paper **SeLa** presented at ICLR 2020 by Asano et al. of the Visual Geometry Group(VGG), University of Oxford improves this approach and achieved state of the art results in various benchmarks.  
![](/images/sela-intro.png){.img-center}
The most interesting side-effect of this method is that we can *auto-generate labels for images in some new domain* and then use those labels to bootstrap our regular workflow of training a supervised model with any architecture. Self-Labelling is very practical for industries and domains with scarce labeled data. Let's understand how it works.

## Solving The Chicken and Egg Problem
At a very high level, the Self-Labelling method works as follows:
 
- Generate the labels and then train a model on these labels
- Generate new labels from the trained model
- Repeat the process

![](/images/sela-chicken-egg-problem.png){.img-center}

> But, how will you generate labels for images in the first place without a trained model? This sounds like the chicken-and-egg problem where if chicken came first, what did it hatch from and if egg came first, who laid the egg?

The solution to the problem is to use a randomly initialized network to bootstrap the first set of image labels. This has been shown to work empirically in the [DeepCluster](https://arxiv.org/abs/1807.05520) paper.  

The authors of DeepCluster used a randomly initialized <span style="color: #51677d;">AlexNet</span> and evaluated it on ImageNet. Since the ImageNet dataset has 1000 classes, if we randomly guessed the classes, we would get an baseline accuracy of <span style="color: #c91212;">1/1000</span> = <span style="color: #c91212;">0.1%</span>. But, a randomly initialized AlexNet was shown to achieve <span style="color: #3fb536;">12%</span> accuracy on ImageNet. This means that a randomly-initialized network possesses some faint signal in its weights.
![](/images/sela-faint-signal.png){.img-center}

Thus, we can use labels obtained from a randomly initialized network to kick start the process which can be refined later.

## Code Implementation
The official implementation of Self-Labelling in PyTorch by the paper authors is available [here](https://github.com/yukimasano/self-label).

## Citation Info (BibTex)
If you found this blog post useful, please consider citing it as:
```
@misc{chaudhary2020SeLa,
  title   = {An Illustrated Guide to Self-Labelling Images},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/03/illustrated-sela}}
}
```

## References
- [Self-labelling via simultaneous clustering and representation learning](https://arxiv.org/abs/1911.05371)
- [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520)