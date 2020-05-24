Title: Self Supervised Representation Learning in NLP
Date: 2020-05-23 16:53
Modified: 2020-05-23 16:53
Category: nlp
Slug: self-supervised-learning-nlp
Summary: An overview of self-supervised pretext tasks in Natural Language Processing
Status: published
Authors: Amit Chaudhary
Cover: /images/nlp-ssl.png

While Computer Vision is making [amazing progress](https://amitness.com/2020/02/illustrated-self-supervised-learning/) on self-supervised learning only in the last few years, self-supervised learning has been a first-class citizen in NLP research for quite a while. Language Models have existed since the 90's even before the phrase "self-supervised learning" was termed. The Word2Vec paper from 2013 popularized this paradigm and the field has rapidly progressed applying these self-supervised methods across many problems.  

At the core of these self-supervised methods lies a framing called "**pretext task**" that allows us to use the data itself to generate labels and use supervised methods to solve unsupervised problems. These are also referred to as "**auxiliary task**" or "**pre-training task**". The representations learned by performing this task can be used as a starting point for our downstream supervised tasks.  
![](/images/nlp-ssl.png){.img-center}  

In this post, I will provide an overview of the various pretext tasks that researchers have designed to learn representations from text corpus without explicit data labeling. The focus of the article will be on the task formulation rather than the architectures implementing them.      

## Self-Supervised Formulations  
## 1. Center Word Prediction  
In this formulation, we take a small chunk of the text of a certain window size and our goal is to predict the center word given the surrounding words.  
![](/images/nlp-ssl-center-word-prediction.gif){.img-center}  
For example, in the below image, we have a window of size of one and so we have one word each on both sides of the center word. Using these neighboring words, we need to predict the center word.    
![](/images/nlp-ssl-cbow-explained.png){.img-center}  
This formulation has been used in the famous "**Continuous Bag of Words**" approach of the [Word2Vec](https://arxiv.org/abs/1301.3781) paper.  

## 2. Neighbor Word Prediction  
In this formulation, we take a span of the text of a certain window size and our goal is to predict the surrounding words given the center word.  
![](/images/nlp-ssl-neighbor-word-prediction.gif){.img-center}  
This formulation has been implemented in the famous "**skip-gram**" approach of the [Word2Vec](https://arxiv.org/abs/1301.3781) paper.  


## 3. Neighbor Sentence Prediction  
In this formulation, we take three consecutive sentences and design a task in which given the center sentence, we need to generate the previous sentence and the next sentence. It is similar to the previous skip-gram method but applied to sentences instead of words.  
![](/images/nlp-ssl-neighbor-sentence.gif){.img-center}  
This formulation has been used in the [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726) paper.

## 4. Auto-regressive Language Modeling  
In this formulation, we take large corpus of unlabeled text and setup a task to predict the next word given the previous words. Since we already know what word should come next from the corpus, we don't need manually-annotated labels.  
![](/images/nlp-ssl-causal-language-modeling.gif){.img-center}   
For example, we could setup the task as left-to-right language modeling by predicting <span style="color: #439f47;">next words</span> given the previous words.  
![](/images/nlp-ssl-causal-language-modeling-steps.png){.img-center}  
We can also formulate this as predicting the <span style="color: #439f47;">previous words</span> given the future words. The direction will be from right to left.  
![](/images/nlp-ssl-causal-rtl.png){.img-center}  

This formulation has been used in many papers ranging from n-gram models to neural network models such as [Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)(Bengio et al., 2003) to [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).

## 5. Masked Language Modeling  
In this formulation, words in a text are randomly masked and the task is to predict them. Compared to the auto-regressive formulation, we can use context from both previous and next words when predicting the masked word.      
![](/images/nlp-ssl-masked-lm.png){.img-center}  
This formulation has been used in the [BERT](https://arxiv.org/abs/1810.04805), [RoBERTa](https://arxiv.org/abs/1907.11692) and [ALBERT](https://arxiv.org/abs/1909.11942) papers. Compared to the auto-regressive formulation, in this task, we predict only a small subset of masked words and so the amount of things learned from each sentence is lower.

## 6. Next Sentence Prediction  
In this formulation, we take two consecutive sentences present in a document and another sentence from a random location in the same document or a different document.  
![](/images/nlp-ssl-nsp-sampling.png){.img-center}  
Then, the task is to classify whether two sentences can come one after another or not.  
![](/images/nlp-ssl-next-sentence-prediction.png){.img-center}  
It was used in the [BERT](https://arxiv.org/abs/1810.04805) paper to improve performance on downstream tasks that requires an understanding of sentence relations such as Natural Language Inference(NLI) and Question Answering. However, later works have questioned its effectiveness.  

## 7. Sentence Order Prediction    
In this formulation, we take pairs of consecutive sentences from the document. Another pair is also created where the positions of the two sentences are interchanged.    
![](/images/nlp-ssl-sop-sampling.png){.img-center}  
The goal is to classify if a pair of sentences are in the correct order or not.  
![](/images/nlp-ssl-sop-example.png){.img-center}  

It was used in the [ALBERT](https://arxiv.org/abs/1909.11942) paper to replace the "Next Sentence Prediction" task.  

## 8. Emoji Prediction  
This formulation was used in the DeepMoji paper and exploits the idea that we use emoji to express the emotion of the thing we are tweeting. As shown below, we can use the emoji present in the tweet as the label and formulate a supervised task to predict the emoji when given the text.  
![](/images/nlp-ssl-deepmoji.gif){.img-center}   
Authors of DeepMoji used this concept to perform pre-training of a model on 1.2 billion tweets and then fine-tuned it on emotion-related downstream tasks like sentiment analysis, hate speech detection and insult detection.  

## Citation Info (BibTex)
If you found this blog post useful, please consider citing it as:
```
@misc{chaudhary2020sslnlp,
  title   = {Self Supervised Representation Learning in NLP},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/05/self-supervised-learning-nlp}
}
```

## References
- Ryan Kiros, et al. ["Skip-Thought Vectors"](https://arxiv.org/abs/1506.06726)
- Tomas Mikolov, et al. ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781)
- Alec Radford, et al. ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Jacob Devlin, et al. ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805)
- Yinhan Liu, et al. ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692)
- Zhenzhong Lan, et al. ["ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"](https://arxiv.org/abs/1909.11942)
- Bjarke Felbo, et al. ["Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm"](https://arxiv.org/abs/1708.00524)  
