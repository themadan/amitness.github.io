Title: A Visual Survey of Data Augmentation in NLP
Date: 2020-05-16 22:22
Modified: 2020-05-16 22:22
Category: nlp
Slug: data-augmentation-for-nlp
Summary: An extensive overview of text data augmentation techniques for Natural Language Processing
Status: published
Authors: Amit Chaudhary
Cover: /images/semantic-invariance-nlp.png

Unlike Computer Vision where using image data augmentation is standard practice, augmentation of text data in NLP is pretty rare. This is because trivial operations for images like rotating an image a few degrees or converting it into grayscale doesn't change its semantics. This presence of semantically invariant transformation is what made augmentation an essential toolkit in Computer Vision research.
![](/images/semantic-invariance-nlp.png){.img-center}

I was curious if there were attempts at developing augmentation techniques for NLP and explored the existing literature. In this post, I will share my findings of the current approaches being used for augmenting text data.  

## Approaches
## 1. Lexical Substitution
This approach tries to substitute words present in a text without changing the gist of the sentence.

- **Thesaurus-based substitution**  
In this technique, we take a random word from the sentence and replace it with its synonym using a Thesaurus. For example, we could use the [WordNet](https://wordnet.princeton.edu/) lexical database for English to look up the synonyms and then perform the replacement. It is a manually curated database with relations between words.
![](/images/nlp-aug-wordnet.png){.img-center}  
[Zhang et al.](https://arxiv.org/abs/1509.01626) used this technique in their 2015 paper "Character-level Convolutional Networks for Text Classification". [Mueller et al.](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023) used a similar strategy to generate additional 10K training examples for their sentence similarity model.  
<br>
NLTK provides a programmatic [access](https://www.nltk.org/howto/wordnet.html) to WordNet. You can also use [TextBlob API](https://textblob.readthedocs.io/en/dev/quickstart.html#wordnet-integration). There is also a database called [PPDB](http://paraphrase.org/#/download) containing millions of paraphrases that you can download and access programmatically.  

- **Word-Embeddings Substitution**  
In this approach, we take pre-trained word embeddings such as Word2Vec, GloVe, FastText, Sent2Vec, and use the nearest neighbor words in the embedding space as the replacement for some word in the sentence. [Jiao et al.](https://arxiv.org/abs/1909.10351) have used this technique with GloVe embeddings in their paper "*TinyBert*" to improve generalization of their language model on downstream tasks. [Wang et al.](https://www.aclweb.org/anthology/D15-1306.pdf) used it to augment tweets needed to learn a topic model.  
![](/images/nlp-aug-embedding.png){.img-center}  
For example, you can replace the word with the 3-most similar words and get three variations of the text.
![](/images/nlp-aug-embedding-example.png){.img-center}  
It's easy to use packages like Gensim to access pre-trained word vectors and get the nearest neighbors. For example, here we find the synonyms for the word 'awesome' using word vectors trained on tweets.  
```python
# pip install gensim
import gensim.downloader as api

model = api.load('glove-twitter-25')  
model.most_similar('awesome', topn=5)
```
You will get back the 5 most similar words along with the cosine similarities.
```python
[('amazing', 0.9687871932983398),
 ('best', 0.9600659608840942),
 ('fun', 0.9331520795822144),
 ('fantastic', 0.9313924312591553),
 ('perfect', 0.9243415594100952)]
```  
  
- **Masked Language Model**  
Transformer models such as BERT, ROBERTA and ALBERT have been trained on a large amount of text using a pretext task called "Masked Language Modeling" where the model has to predict masked words based on the context.  
<br>
This can be used to augment some text. For example, we could use a pre-trained BERT model and mask some parts of the text. Then, we use the BERT model to predict the token for the mask.  
![](/images/nlp-aug-bert-mlm.png){.img-center}
Thus, we can generate variations of a text using the mask predictions. Compared to previous approaches, the generated text is more grammatically coherent as the model takes context into account when making predictions.
![](/images/nlp-aug-bert-augmentations.png){.img-center}  
This is easy to implement with open-source libraries such as transformers by Hugging Face. You can set the token you want to replace with `<mask>` and generate predictions.  
```python
from transformers import pipeline
nlp = pipeline('fill-mask')
nlp('This is <mask> cool')
```

```python
[{'score': 0.515411913394928,
  'sequence': '<s> This is pretty cool</s>',
  'token': 1256},
 {'score': 0.1166248694062233,
  'sequence': '<s> This is really cool</s>',
  'token': 269},
 {'score': 0.07387523353099823,
  'sequence': '<s> This is super cool</s>',
  'token': 2422},
 {'score': 0.04272908344864845,
  'sequence': '<s> This is kinda cool</s>',
  'token': 24282},
 {'score': 0.034715913236141205,
  'sequence': '<s> This is very cool</s>',
  'token': 182}]
```  
However one caveat of this method is that deciding which part of the text to mask is not trivial. You will have to use heuristics to decide the mask, otherwise the generated text will not retain the meaning of original sentence.  


- **TF-IDF based word replacement**  
This augmentation method was proposed by [Xie et al.](https://arxiv.org/abs/1904.12848) in the Unsupervised Data Augmentation paper. The basic idea is that words that have <span style="color: #d52f2f;">low TF-IDF scores</span> are uninformative and thus can be replaced without affecting the ground-truth labels of the sentence.
![](/images/nlp-aug-tf-idf-word-replacement.png){.img-center}  
The words to replace with are chosen from the whole vocabulary that have low TF-IDF scores in the whole document. You can refer to the implementation in the original paper from [here](https://github.com/google-research/uda/blob/master/text/augmentation/word_level_augment.py).



## 2. Back Translation
In this approach, we leverage machine translation to paraphrase a text while retraining the meaning. [Xie et al.](https://arxiv.org/abs/1904.12848) used this method to augment the unlabeled text and learn a semi-supervised model on IMDB dataset with only 20 labeled examples. The method outperformed the previous state-of-the-art model trained on 25,000 labeled examples.

The back-translation process is as follows:  

- Take some sentence (e.g. in English) and translate to another Language e.g. French  
- Translate the french sentence back into English sentence  
- Check if the new sentence is different from our original sentence. If it is, then we use this new sentence as an augmented version of the original text.  
![](/images/nlp-aug-back-translation.png){.img-center}  

You can also run back-translation using different languages at once to generate more variations. As shown below, we translate an English sentence to a target language and back again to English for three target languages: French, Mandarin and Italian. 
![](/images/nlp-aug-backtranslation-multi.png){.img-center}  
This technique was also used in the [1st place solution](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557) for the "Toxic Comment Classification Challenge" on Kaggle. The winner used it for both training-data augmentation as well as during test-time where the predicted probabilities for English sentence along with back-translation using three languages(French, German, Spanish) were averaged to get the final prediction.    

For implementation of back-translation, you can use TextBlob. Alternatively, you can also use Google Sheets and follow the instructions given [here](https://amitness.com/2020/02/back-translation-in-google-sheets/) to apply Google Translate for free.  

## 3. Text Surface Transformation
These are simple pattern matching transformations applied using regex and was introduced by [Claude Coulombe](https://arxiv.org/abs/1812.04718) in his paper.

In the paper, he gives an example of transforming verbal forms from contraction to expansion and vice versa. We can generate augmented texts by applying this.  
![](/images/nlp-aug-contraction.png){.img-center}  
Since the transformation should not change the meaning of the sentence, we can see this can fail in case of expanding ambiguous verbal forms like:
![](/images/nlp-aug-contraction-ambiguity.png){.img-center}  
To resolve this, the paper proposes that we allow ambiguous contractions but skip ambiguous expansion.  
![](/images/nlp-aug-contraction-solution.png){.img-center}  
You can find a list of contractions for the English language [here](https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions).

## 4. Random Noise Injection
The idea of these methods is to inject noise in the text so that the model trained is robust to perturbations.  

- **Spelling error injection**  
In this method, we add spelling error to some random in the sentence. These spelling errors can be added programatically or using a mapping of common spelling errors such as [this list](https://github.com/makcedward/nlpaug/blob/master/model/spelling_en.txt) for English.  
![](/images/nlp-aug-spelling-example.png){.img-center}  

- **QWERTY Keyboard Error Injection**    
This method tries to simulate common errors that happens when typing on a QWERTY layout keyboard due to keys that are very near to each other. The errors are injected based on keyboard distance.  
![](/images/nlp-aug-keyboard-error-example.png){.img-center}  

## Implementation
To apply all the above methods, you can use the python library called [nlpaug](https://github.com/makcedward/nlpaug). It provides a simple and consistent API to apply these techniques.  

## Conclusion  
My takeaway from the literature review is that many of these augmentation methods are very task-specific and their impact on performance have been studied for some particular use-cases only. It would be an interesting research to systematically compare these methods and analyze their impact on performance for many tasks.    

## References
- Qizhe Xie, et al. ["Unsupervised Data Augmentation for Consistency Training"](https://arxiv.org/abs/1904.12848)  
- Claude Coulombe ["Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs"](https://arxiv.org/abs/1812.04718)
- Xiaoqi Jiao, et al. ["TinyBERT: Distilling BERT for Natural Language Understanding"](https://arxiv.org/abs/1909.10351)
- Xiang Zhang, et al. ["Character-level Convolutional Networks for Text Classification"](https://arxiv.org/abs/1509.01626)