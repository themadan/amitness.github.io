Title: Zero-Shot Learning for Text Classification
Date: 2020-05-30 17:27
Modified: 2020-05-30 17:27
Category: nlp
Slug: zero-shot-text-classification
Summary: A visual overview of zero-shot learning for text classification in Natural Language Processing 
Status: draft
Authors: Amit Chaudhary
Cover: /images/zero-shot-paper-idea.png

The recent release of GPT-3 got me interested in the state of zero-shot learning and few-shot learning in NLP. While most of the zero-shot learning research is concentrated in Computer Vision, there has been some interesting developments in the NLP domain as well.  

I am writing a series of blog posts to cover existing research on zero-shot learning in NLP. In this first article, I will cover the paper ["Train Once, Test Anywhere: Zero-Shot Learning for Text Classification"](https://arxiv.org/abs/1712.05972) by Pushp et al. It was the first work to propose a zero-shot learning paradigm for text classification.  


## What is Zero-Shot Learning?
Zero-Shot Learning is the ability to detect classes that the model has never seen during training. It resembles our ability as humans to generalize and identify new things without explicit supervision.  
 
For example, let's say we want to do <span style="color: #546E7A; font-weight: bold;">sentiment classification</span> and <span style="color: #795548; font-weight: bold;">news category</span> classification. Normally, we will train/fine-tune a new model for each dataset. In contrast, with zero-shot learning, you can perform tasks such as sentiment and news classification directly without any task-specific training.  
![](/images/zero-shot-vs-transfer.png){.img-center}  

## Train Once, Test Anywhere
In the paper, the authors propose a simple idea. Instead of classifying texts into X labels, they re-formulate the task as a binary classification to determine if a text and label are related or not.   
![](/images/zero-shot-paper-idea.png){.img-center}   


## References
- Pushpankar Kumar Pushp, et al. ["Train Once, Test Anywhere: Zero-Shot Learning for Text Classification"](https://arxiv.org/abs/1712.05972)
