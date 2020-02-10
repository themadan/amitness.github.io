Title: Illustrated Guide to Applying LSTM in Keras
Date: 2020-02-10 03:00
Modified: 2020-02-10 03:00
Category: nlp
Slug: illustrated-lstm-keras
Summary: An illustrated guide on applying LSTM in Keras
Status: draft
Authors: Amit Chaudhary


From the realm of Dense and Convolution layers, when I first came across sequence modeling architectures such as RNN, GRU and LSTM, I was intrigued by how simple but powerful they were.

However, when it came to implementation using frameworks like Keras, there were  many roadblocks ranging from input shape issues to not understanding how to formulate a problem using LSTM itself. 

Having learnt these things the hard way, I've built a mental model now to easily design architectures and leverage LSTMs on various problems. This post is an attempt to make these mental model concrete and allow others to leverage it. The mental model rests on this core idea of being able to visualize at a high level how the inputs and outputs from these units pass.

I will be explaning the mental model through the lens of Natural Language Processing to make the examples intuitive.

## The Higher Purpose
Let's say we have some text and you want to apply machine learning on it. We know models only understand numeric data, so we need some way to convert it into numeric form. We have word vector techniques to get the numeric form (embeddings) for the words.
![](/images/rnn-necessity.png){.img-center}    
We see above how the sentence as a whole is negative because of present of the word "**not**" before "**good**". We would want our ML models such that previous words are taken into current when processing the current word. Thus, we need a concept of state/memory. That's where RNN shines.

## RNNs in Keras
Let's take a simple example of encoding a sentence using RNN layer in Keras.

![](/images/i-am-groot-sentence.png){.img-center}
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


## Conclusion

## References
- [Keras Documentation](https://keras.io/)