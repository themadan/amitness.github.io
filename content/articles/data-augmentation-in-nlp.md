Title: A Visual Overview of Data Augmentation in NLP
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

## References
- []()  