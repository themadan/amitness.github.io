Title: Identify the Language of Text using Python
Date: 2019-07-15 10:44
Modified: 2019-07-15 10:44
Tags: fasttext, language
Category: nlp
Authors: Amit Chaudhary
Slug: identify-text-language-python
Summary: Learn how to predict the language of a given piece of text using Natural Language Processing.
Status: published


Text language identification is the process of determining the language of a given a piece of text. An example use case is of Chrome offering to translate a webpage when it detects a non-English webpage. In Natural Language Processing, it can be useful to separate out mixed language text in a corpus.

## Facebook's Fasttext library
The popular fasttext library provides pre-trained models for text-based language identification. It can recognize 170 languages, takes less than 1MB of memory and can classify thousands of documents per second. These models were trained on data from Wikipedia, Tatoeba, and SETimes and are released as open source, free to use by everyone. 


![](https://fasttext.cc/img/blog/2017-10-02-blog-post-img1.png)
<p align="center" style="font-style: italic;">Source: [Fasttext Blog](https://fasttext.cc/blog/2017/10/02/blog-post.html)</p>

## Steps
- Install `Fasttext` library
```
pip install fasttext
```   

- There are two versions of the pre-trained models. Choose the model which fits your requirements:
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

- To programmatically convert language symbols back to the language name, you can use `pycountry` package. Install the package using pip.
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
