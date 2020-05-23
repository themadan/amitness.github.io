Title: Self Supervised Representation Learning in NLP
Date: 2020-05-23 16:53
Modified: 2020-05-23 16:53
Category: nlp
Slug: self-supervised-learning-nlp
Summary: An extensive overview of self-supervised pretext tasks in Natural Language Processing
Status: draft
Authors: Amit Chaudhary
Cover: /images/semantic-invariance-nlp.png

While Computer Vision is making [amazing progress](https://amitness.com/2020/02/illustrated-self-supervised-learning/) on self-supervised learning only in the last few years, self-supervised learning has been a first-class citizen in NLP research for quite a while. Language Models have existed since the 90's even before the phrase "self-supervised learning" was termed. The Word2Vec paper from 2013 really popularized this paradigm and the field has rapidly progressed applying these self-supervised methods across many problems.  

At the core of these self-supervised methods lies a framing called "**pretext task**" that allows us to use the data itself to generate labels and use supervised methods to solve unsupervised problems. These are also referred to as "**auxiliary tasks**" or "**pre-training tasks**".

In this post, I will provide an overview of the various problem formulations that researchers have designed to learn representations from text corpus without explicit data labeling. The focus will be on the formulation rather than the architecture.  

## Problem Formulations for NLP    
## 1. Auto-regressive Language Modeling  
In this formulation, we take large corpus of unlabeled text and setup a task to predict the next word given the previous words. Since we already know what word should come next from the corpus, we don't need manually-annotated labels.  
![](/images/nlp-ssl-causal-language-modeling.gif){.img-center}   
For example, we could setup the task as left-to-right language modeling by predicting next words given the previous words.  
![](/images/nlp-ssl-causal-language-modeling-steps.png){.img-center}  
We can also formulate this as predicting the previous words given the future words. 
![](/images/nlp-ssl-causal-rtl.png){.img-center}  

This formulation has been used in many papers ranging from n-gram models to neural network models like GPT.

## 2. Masked Language Modeling  
In this formulation, words in a text are randomly masked and the task is to predict them. Compared to auto-regressive formulation, we can use context from both previous and next words when predicting the masked word.      
![](/images/nlp-ssl-masked-lm.png){.img-center}  

## 3. Next Sentence Prediction  
![](/images/nlp-ssl-nsp-sampling.png){.img-center}  
![](/images/nlp-ssl-next-sentence-prediction.png){.img-center}  


## 4. Sentence Order Prediction    
![](/images/nlp-ssl-sop-sampling.png){.img-center}  
![](/images/nlp-ssl-sop-example.png){.img-center}  


## Citation Info (BibTex)
If you found this blog post useful, please consider citing it as:
```
@misc{chaudhary2020sslnlp,
  title   = {Self Supervised Learning in NLP},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/05/self-supervised-learning-nlp}
}
```

## References
- XYZ, et al. ["#"](https://arxiv.org/abs/1904.12848)  
