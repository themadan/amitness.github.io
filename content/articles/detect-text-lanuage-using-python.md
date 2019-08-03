Title: Identify the Language of Text using Python
Date: 2019-07-15 10:44
Modified: 2019-08-03 12:44
Tags: fasttext, language
Category: nlp
Authors: Amit Chaudhary
Slug: identify-text-language-python
Summary: Learn how to predict the language of a given piece of text using Natural Language Processing.
Status: published


Text Language Identification is the process of predicting the language of a given a piece of text. You might have encountered it when Chrome shows a popup to translate a webpage when it detects that the content is not in English. Behind the scenes, Chrome is using a model to predict the language of text used on a webpage.

![Google Translate Popup on Chrome](/images/google_translate_popup.png)

When working with a dataset for NLP,  the corpus may contain a mixed set of languages. Here, language identification can be useful to either filter out a few languages or to translate the corpus to a single language and then use for your downstream tasks.

In this post, I will demonstrate how to use the Fasttext library for language identification.

## Facebook's Fasttext library
![Fasttext Logo](/images/fastText_logo.png) 
 
[Fasttext](https://fasttext.cc/) is an open-source library in Python for word embeddings and text classification. It is built for production use rather than research and hence is optimized for performance and size. It extends the [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) model with ideas such as using [subword information](https://arxiv.org/abs/1607.04606) and [model compression](https://arxiv.org/abs/1612.03651).


For our purpose of language identification, we can use the pre-trained models provided by fastText. The model was trained on a dataset drawn from [Wikipedia](https://www.wikipedia.org/), [Tatoeba](https://tatoeba.org/eng/), and [SETimes](http://nlp.ffzg.hr/resources/corpora/setimes/). The basic idea is a prepare a training data of (text, language) pairs and then train a classifier on it.
 

![Language Training Data Example](/images/lang_training_data.png) 

From the benchmark on their blog, we can see that the pre-trained models are better than [langid.py](https://github.com/saffsd/langid.py), another popular language identification tool. Fasttext has better accuracy and also the inference time is very fast. It supports a wide variety of languages including French, German, English, Spanish, Chinese.

![](https://fasttext.cc/img/blog/2017-10-02-blog-post-img1.png)
<p align="center" style="font-style: italic;">Source: [Fasttext Blog](https://fasttext.cc/blog/2017/10/02/blog-post.html)</p>

## Steps
- Install the `Fasttext` library using pip.
```
pip install fasttext
``` 

- There are two versions of the pre-trained models. Choose the model which fits your memory and space requirements:
 - [lid.176.bin](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin): faster and slightly more accurate but 126MB in size
 - [lid.176.ftz](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz): a compressed version of the model, with a file size of 917kB

- Download the pre-trained model from Fasttext to some location. You'll need to specify this location later in the code. In our example, we download it to the /tmp directory. 
```
wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

- Now, we import fasttext and then load the model from the pretrained path we downloaded earlier.
```python
import fasttext

PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)
```

- Let's take an example sentence in French which means 'I eat food'. To detect it's language, just pass a list of sentences to the predict function. The sentences should be in the UTF-8 format.

![YouTube HTML5 Player](/images/french_to_english_translation.png) 


```python
sentences = ['je mange de la nourriture']
predictions = model.predict(sentences)
print(predictions)

# ([['__label__fr']], array([[0.96568173]]))
```
- The model returns back two tuples back. One of them is an array of language labels and the other is the confidence for each sentence. Here 'fr' is the ISO 639 code for French. The model is 96.56% confident that the language is French.

- Fasttext can detect 170 languages and returns one among these ISO codes. You can refer to the page on [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) codes to find language for each symbol.
```
af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh
```

- To programmatically convert language symbols back to the language name, you can use [pycountry](https://pypi.org/project/pycountry/) package. Install the package using pip.
```python
pip install pycountry
```

- Now, pass the symbol to pycountry and you will get back the language name.
```python
from pycountry import languages

language_name = languages.get(alpha_2='fr').name
print(language_name)
# french
```