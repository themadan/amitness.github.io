Title: Zero-Shot Learning for Text Classification
Date: 2020-05-30 17:27
Modified: 2020-05-30 17:27
Category: nlp
Slug: zero-shot-text-classification
Summary: A visual overview of zero-shot learning for text classification in Natural Language Processing 
Status: draft
Authors: Amit Chaudhary
Cover: /images/zero-shot-paper-idea.png

The recent release of GPT-3 got me interested in the state of zero-shot learning and few-shot learning in NLP. While most of the zero-shot learning research is concentrated in Computer Vision, there has been some interesting work in the NLP domain as well.  

I will be writing a series of blog posts to cover existing research on zero-shot learning in NLP. In this first post, I will explain the paper ["Train Once, Test Anywhere: Zero-Shot Learning for Text Classification"](https://arxiv.org/abs/1712.05972) by Pushp et al. This paper from December 2017 was the first work to propose a zero-shot learning paradigm for text classification.  


## What is Zero-Shot Learning?
Zero-Shot Learning is the ability to detect classes that the model has never seen during training. It resembles our ability as humans to generalize and identify new things without explicit supervision.  
 
For example, let's say we want to do <span style="color: #546E7A; font-weight: bold;">sentiment classification</span> and <span style="color: #795548; font-weight: bold;">news category</span> classification. Normally, we will train/fine-tune a new model for each dataset. In contrast, with zero-shot learning, you can perform tasks such as sentiment and news classification directly without any task-specific training.  
![](/images/zero-shot-vs-transfer.png){.img-center}  

## Train Once, Test Anywhere
In the paper, the authors propose a simple idea for zero-shot classification. Instead of classifying texts into X classes, they re-formulate the task as a binary classification to determine if a text and a class are related or not.   
![](/images/zero-shot-paper-idea.png){.img-center}   

Let's understand their formulation in more details now.    
## 1. Data Preparation  
The authors crawled 4.2 million <span style="color: #7E57C2; font-weight: bold;">news headlines</span> from the web and used the <span style="color: #795548; font-weight: bold;">SEO tags</span> for the news article as the <span style="color: #795548; font-weight: bold;">labels</span>. After crawling, they got total <span style="color: #795548; font-weight: bold;">300,000 unique tags</span> as the labels. We can see how troublesome it would have been if we had to train a supervised model on <span style="color: #795548; font-weight: bold;">300,000 classes</span>. 
![](/images/zero-shot-data-crawling.png){.img-center}  
Each <span style="color: #7E57C2; font-weight: bold;">headline</span> was truncated to 28 words and anything shorter was padded.  

## 2. Word Embedding   
The paper uses word2vec pre-trained on Google News as the word embeddings for both the sentences as well as the labels.  

## 3. Architecture     
The paper proposes three different architecture for their formulation:
## A. Architecture 1  
This architecture takes the mean of word embeddings in the sentence as the sentence embedding and concat it with the label embedding. This vector is then passed through a full connected layer to classify if the sentence and label are related or not.      


## References
- Pushpankar Kumar Pushp, et al. ["Train Once, Test Anywhere: Zero-Shot Learning for Text Classification"](https://arxiv.org/abs/1712.05972)
