Title: Visual Paper Summary: ALBERT (A Lite BERT)
Date: 2020-02-02 20:00
Modified: 2020-02-02 20:00
Category: nlp
Slug: albert-visual-summary
Summary: Learn how to integrate and finetune tensorflow-hub modules in Tensorflow 2.0
Status: draft
Authors: Amit Chaudhary

The release of BERT model by google is considered as the "imagenet moment" for NLP and start of a new era for NLP.

Let's recap BERT for a while.


# Quick Recap:  BERT
Consider a sentence given below. As humans, when we encounter the word "**apple**", we could: 

- Associate the word "apple" to our mental representation of the fruit "apple"  
- Associate "apple" to the fruit rather than the company based on the context  
- Understand the big picture that "*he ate an apple*"  
- Distinguish between characters "a", word and sentences

![](/images/nlp-representation-learning.png){.img-center}    

The basic premise of latest developments in NLP is to give machines the ability to learn such representations.

In 2018, Google release BERT that attempted to solve this task based on a few novel ideas:
### 1. Masked Language Modeling
Language modeling basically involves predicting the word given its context to learn representation. Tradionally, this involved predicting the next word in sentence given previous words.
![](/images/nlp-language-model-1.png){.img-center}  

BERT introduced a **masked language model** objective, in which we randomly mask words in document and try to predict them based on context.
![](/images/bert-masked-language-model.png){.img-center}  
<p class="has-text-centered">
Credits: [Marvel Studios on Giphy](https://giphy.com/stickers/marvelstudios-oh-thanos-snapped-TfjfIgE9YUgdyz8V1J)
</p>

### 2. Next Sentence Prediction
Idea here is to detect whether two sentences are coherent when placed one after another or not.
![](/images/bert-nsp.png){.img-center}  

For this, BERT takes consecutive sentences from training data as a positive example. For negative example, a sentence is taken and a random sentence from another document is placed next to it.

### 3. Transformer Architecture
To solve the above two tasks, BERT uses stacked layers of transformer blocks. We get vectors of size 768 for each word when passing through the layers to capture the meaning of the word.
![](/images/bert-blocks.png)
Jay Alammar has an [excellent post](http://jalammar.github.io/illustrated-bert/) that illustrates the internals of BERT in more depth.

## Problems with BERT
Even though BERT gave state of art results on many NLP leaderboards, there are issues with it that are highlighted by the ALBERT paper:

1. **Memory Limitation and Communication Overhead:**  
    ![](/images/bert-heavy-on-gpu.png){.img-center}  
    BERT Large has billions of parameters contributed by its large 24 hidden layers, large units and attention heads in each layer. This makes it difficult to take large BERT models as it is and use for real-world application where restricted compute requirements play a major role.
    Due to memory limitations of currently available GPU/TPUs, we can't keep doubling the size of models and expect it to work.  
    
    One way to train solve the limited memory issue would be to use distributed training. But, for such configuration, the training time is slowed due to the large number of parameters of the model which adds communication overhead.
    
2. **Model Degradation**  
    Recent trend in the NLP community is that models size are getting larger and larger while leading to state-of-the-art performance on the various datasets.   
      
    In the ALBERT paper, the author performed an interesting experiment. If larger models lead to better performance, why not double the hidden layer units of the largest available BERT model, the BERT-large with 1024 units to BERT-xlarge with 2048 units. Contrary to expectation, the larger model actually performs worse than the BERT-large model on both Language Model accuracy as well as performance on the RACE dataset.
    ![](/images/bert-doubled-performance-race.png)
    
    From the plots given in the original paper, we can see how the performance degrades. BERT-xlarge is performing worse than BERT-large even though it is larger in size. This shows how increasing the size of model doesn't always lead to better performance.
    ![](/images/bert-xlarge-vs-bert-large.png)
    <p class="has-text-centered">
        Credits: ALBERT paper
    </p>

## Solutions from ALBERT
ALBERT solves the problems with BERT with a few interesting ideas:  

1. **Factorized embedding parametrization**   
    In BERT, the word piece embeddings size is linked to the hidden layer size of the transformer blocks. In BERT, the authors have used a vocabulary size of 30000 to learnt word-piece embeddings from the One Hot Encoding Representations. This embeddings are directly projected to the hidden space of the hidden layer.

    So, consider a vocabulary of size V, word-piece embeddings denoted by E and hidden layer size H. If we want to increase the vocabulary, we need to increase the hidden layer size of the blocks as well. And, conversely, if we increase hidden layer size, we new to add a new dimensions to each embedding which is already sparse. 
    
    So, the complexity is O(V*E). This problem is prevalent with even later improvements to BERT in XLNET and ROBERTA as well.
    
    ALBERT solves this problem by factorizing the large vocabulary embedding matrix into two smaller matrices. This separates the size of the hidden layers from the size of the vocabulary embeddings. And this allows us to grow the hidden size without significantly increasing the parameter size 
    of the vocabulary embeddings.
    
    We project the One Hot Encoding vector into the lower dimension embedding space of E and then this embedding space into the hidden space.
    
    Thus, the complexity decreases from O(V\*E) to O(V\*E) + O(E\*H)
    
2. **Cross-layer parameter sharing**  
    The next thing is the depth of the BERT model. The BERT large model has 24 layers compared to it's 12-layer base model. As the depth of layers increases, the number of parameters also grows exponentially.
    To prevent this problem, ALBERT uses the concept of cross-layer parameter sharing. We can either share the FFN layer only, the attention parameters only or share all the parameters between the layers.
    
    ![](/images/albert-parameter-sharing.png)

3. **Sentence-Order prediction (SOP)**  
    BERT introduced a binary classification loss called "Next Sentence Prediction". As explained above, here we took two segments that appear consecutively from training corpus and also a random pair of segment from different document as negative examples. This was specifically created to improve performance on Natural Language Inference where we reason about sentence pair.
    
    Paper like ROBERTA and XLNET have shed light on the ineffectiveness of NSP and found it's impact on the downstream tasks unreliable. On eliminating it, the performance across several tasks have improved.
    
    ALBERT proposes a conjecture that NSP is not a difficult task compared to masked language modeling. It mixes topic prediction and coherence prediction. Topic prediction is easy to learn compared to coherence prediction. Also, topic prediction overlaps with what is learned through the masked language model loss.
    
    So, it proposes an alternative task called Sentence Order Prediction loss. It avoids the topic prediction task and only models inter-sentence coherence.
    
    In this we take two consecutive segments from same document as positive example and same segment with order swapped as a negative example.
    ![](/images/sentence-order-prediction.png)
    
    
    The forces the model to learn finer-grained distinction about discourse-level coherence properties.
    
    Results shows that it improves performance on downstream multi-sentence encoding tasks (SQUAD 1.1, 2.0, MNLI, SST-2, RACE).
    
    [TODO: Add figure from section 4.6]
    
    NSP cannot solve SOP task, only slightly better than random baseline of 52%, but SOP can solve the NSP task
    
    Also, it is convicing evidence that SOP leads to better learning representation.
    
    - model inter-sentence coherence
    - like fasttext and infersent

## Objective Results
- vs BERT-large: 18x fewer parameters
- trained 1.7x faster
- regularization and generalization
- empirical evidence:
    - scale much better
    - SOTA on GLUE, RACE and SQUAD
    - RACE -> 44.1%(2017), 83.2% -> 89.4% (45.3% imporvement)
- RACE accuracy: 89.4%, GLUE benchmark: 89.4, F1 score SQUAD 2.0: 92.2

## Existing solutions: (incorporate above in description)
- Megatron-LM: model parallelism to solve memory limitation: https://arxiv.org/abs/1909.08053
- problem: 


## References
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf)