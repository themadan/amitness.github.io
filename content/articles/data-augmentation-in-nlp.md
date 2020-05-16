Title: A Visual Survey of Data Augmentation in NLP
Date: 2020-05-16 14:06
Modified: 2020-05-16 14:06
Category: nlp
Slug: data-augmentation-for-nlp
Summary: An overview of data augmentation techniques for NLP
Status: draft
Authors: Amit Chaudhary

Unlike Computer Vision where using image data augmentation is a standard practice, augmentation of text data in NLP is pretty rare. This is because trivial operations for images like rotating an image a few degrees or converting it into grayscale doesn't change its semantics. This presence of semantically invariant transformation is what made augmentation an essential toolkit in Computer Vision research.
![](/images/semantic-invariance-nlp.png){.img-center}

I was curious if there were attempts at developing augmentation techniques for NLP and explored the the existing literature. In this post, I will share my findings and give an overview of current approaches being used for augmenting text data.

## Approaches
## 1. Lexical Substitution
This approach tries to substitute words present in a text without changing the gist of the sentence.

- **Thesarus-based substitution**  
In this technique, we take a random word from the sentence and replace it with its synonym using a thesarus. For example, we could use the [WordNet](https://wordnet.princeton.edu/) lexical database for English to look up the synonyms and then perform the replacement. It is a manually curated database with relations between words.
![](/images/nlp-aug-wordnet.png){.img-center}
NLTK provides a programmatic [access](https://www.nltk.org/howto/wordnet.html) to WordNet. You can also use [TextBlob API](https://textblob.readthedocs.io/en/dev/quickstart.html#wordnet-integration).

- **Word-Embeddings Substitution**  
In this approach, we take pre-trained word embeddings such as Word2Vec, GloVe, FastText, Sent2Vec and use the nearest neighbor words in the embedding space as the replacement for some word in the sentence.
![](/images/nlp-aug-embedding.png){.img-center}  
For example, you can replace with the 3-most similar words and get three variations of the text.
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
Transformer models such as BERT, ROBERTA and ALBERT have been trained on large amount of text based on a pretext task called "Masked Language Modeling" where the model has to predict masked words based on the context.  
<br>
This can be used to augment some text. For example, we could use a pre-trained BERT model and mask some part of the text. Then, we use the BERT model to predict the token for the mask.  
![](/images/nlp-aug-bert-mlm.png){.img-center}
Thus, we can generate variations of a text using the mask predictions. Compared to previous approaches, the generated text is more gramatically consistent as the model takes context into account when making predictions.
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

## 2. Back Translation
In this approach, we leverage machine translation to paraphrase a text while retraining the meaning. The way it works is:  

- Take some sentence (e.g. in English) and translate to another Language e.g. French  
- Translate the french sentence back into English sentence  
- Check if the new sentence is different from our original sentence. If it is, then we use this new sentence as augmented version of original text.  
![](/images/nlp-aug-back-translation.png){.img-center}  

You can also run back-translation for multiple languages at once to generate many variations. As shown below, we translate English sentence to a target language and back again to English for 3 target languages: French, Mandarin and Italian.  
![](/images/nlp-aug-backtranslation-multi.png){.img-center}  

For implementation, you can use TextBlob. For free translation, you can use Google Sheets and follow the instructions given [here](https://amitness.com/2020/02/back-translation-in-google-sheets/).