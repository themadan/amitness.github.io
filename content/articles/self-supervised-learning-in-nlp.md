Title: Self Supervised Learning in NLP
Date: 2020-05-23 16:53
Modified: 2020-05-23 16:53
Category: nlp
Slug: self-supervised-learning-nlp
Summary: An extensive overview of self-supervised pretext tasks in Natural Language Processing
Status: draft
Authors: Amit Chaudhary
Cover: /images/semantic-invariance-nlp.png

While Computer Vision has had some [amazing progress](https://amitness.com/2020/02/illustrated-self-supervised-learning/) on self-supervised learning only in the last few years, it has been a first-class citizen in NLP research for quite a while. Language Models have existed since the 90's even before the phrase "self-supervised learning" was termed and popularized. The Word2Vec paper from 2013 popularized this paradigm and the field has rapidly progressed applying these self-supervised methods across many problems.  

At the core of these self-supervised methods lies a framing called "**pretext task**" that allows us to use the data itself to generate labels and use supervised methods to solve unsupervised problems. These are also known as "**auxiliary tasks**" and "**pre-training tasks**".

In this post, I will provide an overview of the various pretext tasks that researchers have designed to learn representations from text corpus in the wild without explicit data labeling.  
 


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
