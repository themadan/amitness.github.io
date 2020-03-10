Title: Illustrated Guide to Recurrent Layers in Keras
Date: 2020-03-07 03:00
Modified: 2020-03-07 03:00
Category: nlp
Slug: illustrated-lstm-keras
Summary: Understand how to use Recurrent Layers in Keras with diagrams.
Status: draft
Authors: Amit Chaudhary


From the realm of simple feed-forward networks, when I first came across sequence modeling architectures such as RNN, GRU and LSTM, I was intrigued by how simple but powerful they were.

However, when it came to leveraging them using frameworks like Keras, there were many roadblocks ranging from not understanding how to formulate a problem using the available layers to input shape issues. 

Having learnt these things the hard way, I've built a mental model now to design architectures with sequence model on various problems. This post is an attempt to make these mental model concrete for others. The mental model rests on this idea of being able to visualize how the inputs and outputs pass from these units.

In this post, I will be explaining the mental model through the lens of Natural Language Processing to make the examples intuitive.

## The Core Purpose
Let's say we have some text and you want to apply machine learning on it. We know models only understand numeric data, so we need some way to convert it into numeric form. We can leverage word vector technique to get the numeric form (embeddings) for the words.
![Necessity of RNN](/images/rnn-necessity.png){.img-center}    
We see above how the sentence as a whole is negative because of the presence of the word "**not**" before "**good**". We would want our ML models such that previous words are taken into account when processing the current word. Thus, we need a concept of state/memory. That's where RNN shines.

## RNNs in Keras
Let's take a simple example of encoding a sentence using RNN layer in Keras.

![I am Groot Sentence](/images/i-am-groot-sentence.png){.img-center}
<p class="has-text-centered">
Credits: Marvel Studios
</p>

To use this sentence in a RNN, we need to convert it into numeric form. Let's consider we have a simplified embedding algorithm that takes a word and convert each word into 2 numbers.

![](/images/i-am-groot-embedding.png){.img-center}

Now, to pass these words into a RNN, we treat each word as time-step and the embedding as it's features. Let's build a RNN layer to pass these into
```python
model = Sequential()
model.add(SimpleRNN(4, input_shape=(3, 2)))
```

To understand what each parameter means, refer to the figure below and how each parameter is linked to that.
![](/images/rnn-default-keras.png)  

- **input_shape=(3, 2)**:  
    - We have 3 words "**I**", "**am**", "**groot**". Thus, number(time-steps) = number(words) = 3. The RNN block unfolds 3 times, and so we see 3 blocks in the figure.
    - Each word is represented by embedding of size 2
    - So input_shape=(3, 2).
- **SimpleRNN(`4, ...`)**:  
    - This denotes the number of units in the hidden layer
    - Here we set 4 hidden units
    - So, in the figure, we see how a hidden state of size 4 is passed between the RNN blocks
    - For the first block, since there is no previous output, so previous hidden state is set to **[0, 0, 0, 0]**

## Stacking Layers
```python
model = Sequential()
model.add(SimpleRNN(4, input_shape=(3, 2), return_sequences=True))
model.add(SimpleRNN(4))
```

![](/images/rnn-stacked.png){.img-center}

## References
- [Recurrent Layers - Keras Documentation](https://keras.io/layers/recurrent/)