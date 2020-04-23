Title: A Visual Guide to Recurrent Layers in Keras
Date: 2020-04-23 02:22
Modified: 2020-04-23 02:22
Category: nlp
Slug: recurrent-layers-keras
Summary: Understand how to use Recurrent Layers like RNN, GRU and LSTM in Keras with diagrams.
Status: published
Authors: Amit Chaudhary
Cover: /images/rnn-default-keras.png

Keras provides a powerful abstraction for recurrent layers such as RNN, GRU and LSTM for Natural Language Processing. When I first started learning about them from the documentation, I couldn't clearly understand how to prepare input data shape, how various attributes of the layers affect the outputs and how to compose these layers with the provided abstraction.

Having learned it through experimentation, I wanted to share my understanding of the API with visualizations so that it's helpful for anyone else having troubles.

## Single Output
Let's take a simple example of encoding the meaning of a whole sentence using a RNN layer in Keras.

![I am Groot Sentence](/images/i-am-groot-sentence.png){.img-center}
<p class="has-text-centered">
Credits: Marvel Studios
</p>

To use this sentence in a RNN, we need to first convert it into numeric form. We could either use one-hot encoding, pretrained word vectors or learn word embeddings from scratch. For simplicity, let's assume we used some word embedding to convert each word into 2 numbers.

![](/images/i-am-groot-embedding.png){.img-center}

Now, to pass these words into a RNN, we treat each word as time-step and the embedding as it's features. Let's build a RNN layer to pass these into
```python
model = Sequential()
model.add(SimpleRNN(4, input_shape=(3, 2)))
```

![](/images/rnn-default-keras.png){.img-center}  
As seen above, here is what the various parameters means and why they were set as such:  

- **input_shape=(<span style="color: #9e74b3;">3</span>, <span style="color: #5aa397;">2</span>)**:  
    - We have <span style="color: #9e74b3;font-weight: bold;">3</span> words: <span style="color: #9e74b3;font-weight: bold;">I</span>, <span style="color: #9e74b3;font-weight: bold;">am</span>, <span style="color: #9e74b3;font-weight: bold;">groot</span>. So, number of time-steps is 3. The RNN block unfolds 3 times, and so we see 3 blocks in the figure.
    - For each word, we pass the <span style="color: #5aa397;font-weight: bold;">word embedding</span> of size <span style="color: #5aa397;font-weight: bold;">2</span> to the network.
- **SimpleRNN(<span style="color: #84b469;">4</span>, ...)**:  
    - This means we have <span style="color: #84b469; font-weight: bold;">4 units</span> in the hidden layer.
    - So, in the figure, we see how a <span style="color: #84b469; font-weight: bold;">hidden state of size 4</span> is passed between the RNN blocks
    - For the first block, since there is no previous output, so previous hidden state is set to **[0, 0, 0, 0]**

Thus for a whole sentence, we get a vector of size 4 as output from the RNN layer as shown in the figure. You can verify this by printing the shape of output from the layer.
```python
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN

x = tf.random.normal((1, 3, 2))

layer = SimpleRNN(4, input_shape=(3, 2))
output = layer(x)

print(output.shape)
# (1, 4)
```
As seen, we create a random batch of input data with 1 sentence having 3 words and each word having an embedding of size 2. After passing through the LSTM layer, we get back representation of size 4 for that one sentence.

<article class="message is-info">
  <div class="message-body">
This can be combined with a Dense layer to build an architecture for something like sentiment analysis or text classification.
<div class="highlight"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">Sequential</span><span class="p">()</span>
<span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">SimpleRNN</span><span class="p">(</span><span class="mi">4</span><span class="p">,</span> <span class="n">input_shape</span><span class="o">=</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">)))</span>
<span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="mi">1</span><span class="p">))</span>
</pre></div>
  </div>
</article>


## Multiple Output
Keras provides a `return_sequences` parameter to control output from the RNN cell. If we set it to `True`, what it means is that the output from each unfolded RNN cell is returned instead of only the last cell.
```python 
model = Sequential()
model.add(SimpleRNN(4, input_shape=(3, 2), 
                    return_sequences=True))
```

![](/images/rnn-return-sequences.png){.img-center}

As seen above, we get an <span style="color: #49a4aa; font-weight: bold;">output vector</span> of size  <span style="color: #49a4aa; font-weight: bold;">4</span> for each word in the sentence. 

This can be verified by the below code where we send one sentence with 3 words and embedding of size 2 for each word. As seen, the layer gives us back 3 outputs with a vector of size 4 for each word.

```python
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN

x = tf.random.normal((1, 3, 2))

layer = SimpleRNN(4, input_shape=(3, 2), return_sequences=True)
output = layer(x)

print(output.shape)
# (1, 3, 4)
```

## TimeDistributed Layer
Suppose we want to recognize entities in a text. For example, in our text "I am <span style="color: #4a820d;">Groot</span>", we want to identify <span style="color: #4a820d;">"Groot"</span> as a <span style="color: #4a820d;">name</span>.
![](/images/keras-groot-ner.png){.img-center}

We have already seen how to get output for each word in the sentence in previous section. Now, we need some way to apply classification on the output vector from the RNN cell on each word. For simple cases such as text classification, you know how we use `Dense()` layer with `softmax` activation as the last layer.  

Similar to that, we can apply <span style="color: #5fb9e0; font-weight: bold;">Dense()</span> layer on <span style="color: #49a4aa; font-weight: bold;">multiple outputs</span> from the RNN layer through a wrapper layer called TimeDistributed(). It will apply the <span style="color: #5fb9e0; font-weight: bold;">Dense</span> layer on <span style="color: #49a4aa; font-weight: bold;">each output</span> and give us class probability scores for the entities. 

```python 
model = Sequential()
model.add(SimpleRNN(4, input_shape=(3, 2), 
                    return_sequences=True))
model.add(TimeDistributed(Dense(4, activation='softmax')))
```

![](/images/keras-time-distributed.png){.img-center}


As seen, we take a 3 word sentence and classify output of RNN for each word into 4 classes using <span style="color: #5fb9e0; font-weight: bold;">Dense layer</span>. These classes can be the entities like name, person, location etc.

## Stacking Layers
We can also stack multiple recurrent layers one after another in Keras.
```python
model = Sequential()
model.add(SimpleRNN(4, input_shape=(3, 2), return_sequences=True))
model.add(SimpleRNN(4))
```

We can understand the behavior of the code with the following figure:
![](/images/rnn-stacked.png){.img-center}

<article class="message is-info">
  <div class="message-header">
    <p>Insight: Why do we usually set return_sequences to True for all layers except the final?</p>  
  </div>
  <div class="message-body">
<p>Since the second layer needs inputs from the first layer, we set return_sequence=True for the first SimpleRNN layer. For the second layer, we usually set it to False if we are going to just be doing text classification. If out task is NER prediction, we can set it to True in the final layer as well.</p>
  </div>
</article>


## References
- [Recurrent Layers - Keras Documentation](https://keras.io/layers/recurrent/)