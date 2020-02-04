Title: Visual Paper Summary: ALBERT (A Lite BERT)
Date: 2020-02-02 20:00
Modified: 2020-02-02 20:00
Category: nlp
Slug: albert-visual-summary
Summary: Learn how to integrate and finetune tensorflow-hub modules in Tensorflow 2.0
Status: published
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

The basic premise of latest developments in NLP is to give machines the ability to do so.

In 2018, Google release BERT model that tried to solve this task based on a few key ideas:
### 1. Masked Language Modeling
Language modeling basically involves predicting the word given its context to learn representation. Tradionally, this involved predicting the next word in sentence given previous words.
![](/images/nlp-language-model-1.png){.img-center}  

BERT introduced a **masked language model** objective, in which we randomly mask words in document and try to predict them based on context.
![](/images/bert-masked-language-model.png){.img-center}  

### 2. Next Sentence Prediction
Idea here is to detect whether two sentences are coherent when placed one after another or not.
![](/images/bert-nsp-1.png){.img-center}  

For this, BERT takes consecutive sentences from training data as a positive example. For negative example, a sentence is taken and a random sentence from another document is placed next to it.

### 3. Transformer Architecture
To solve the above two tasks, BERT uses stacked layers of transformer blocks. We get vectors of size 768 for each word when passing through the layers to capture the meaning of the word.
![](/images/bert-blocks.png)
Jay Alammar has an [excellent post](http://jalammar.github.io/illustrated-bert/) that illustrates the internals of BERT in more depth.

## Problems with BERT
Even though BERT gave state of art results on many NLP leaderboards, there were issues when deploying it for real world applications.

1. **Memory Limitation:**  
![](/images/bert-heavy-on-gpu.png){.img-center}  
BERT Large has billions of parameters due to a large hidden size, 24 hidden layers and more attention heads. Due to memory limitations of currently available GPU/TPUs, we can't keep doubling the size of models and expect it to work. Also, in the real world applications, model size and inference speed are key components of selecting a model.


## Problem
- large model size ++ in pretraining 
-> how: large hidden size, more hidden layers, more attentions heads
-> improved performance on downstream task
- 1. Memory limitation: ++ harder due to GPU/TPU limit:  hundreds of millions or even billions of parameters
- 2. Communication overhead:
--> longer training time: communication overhead in distributed training depend on number of parameters

- 3. Model degradation?
---> BERT-large * 2 -> BERT-xlarge
---> BERT-large: 1024 -> 2048 -> worse performance
---> WOrse performance on RACE

## Existing solutions:
- Megatron-LM: model parallelism to solve memory limitation: https://arxiv.org/abs/1909.08053
- problem: 


## Solution
- solves all 3 problems: reduce memory, increase training speed

2 parameter reduction techniques:
1. Factorized embedding parameterization
why: BERT, XLNET, Roberta: wordpiece embedding E = H

V = 30000

- V*E
if E = H, for V to be large, H has to be large

instead of: projecting OHE -> hidden space of size H
albert: O(V*E) + O(E*H), OHE -> lower dim embedding space of E -> then to hidden space

- large vocabulary embedding matrix into two small matrices
- separate the size of the hidden layers from the size of vocabulary embedding
- grow the hidden size without significantly increasing the parameter size of the vocabulary embeddings

2. cross-layer parameter sharing
- prevents the parameter from growing with the depth of the network
- share: FFN, attention parameters, share all params

3. self-supervised loss for sentence-order prediction (SOP)
What is NSP:
- binary classification loss
- predict whether two segments appear consecutively
- taking consecutive segments from training corpus
- negative by pairing segment from different document
- improve performance for NLI: reasoning about sentence pair

Why removed: - ineffectiveness of NSP -> why?
- ROBERTA and XLNET found impact unreliable and eliminated it, improved performance across several tasks
- ALBERT says:  
- > lack of difficulty as a task, as compared to MLM
-> NSP mixes topic prediction + coherence prediction
-> topic prediction easy to learn compared to coherence prediction
- topic prediction overlaps with what is learned in MLM loss

ALBERTA proposal
- SOP loss
- avoid topic prediction
- only model inter-sentence coherence
- positive example: two consecutive segments from same document
- negative example: same segment with order swapped
- forces model to learn finer-grained distinction about discourse-level coherence properties

improves performance on downstream multi-sentence encoding tasks(given in section 4.6): see section 4.6
- NSP cannot solve SOP : random baseline 52%
- SOP can solve NSP:

Additionally, although we have convincing
evidence that sentence order prediction is a more consistently-useful learning task that leads to better
language representations, we hypothesize that there could be more dimensions not yet captured by
the current self-supervised training losses that could create additional representation power for the
resulting representations.

- model inter-sentence coherence
- help downstream tasks with multi-sentence inputs -> what are those?
- like fasttext and infersent -> 

## Results
- vs BERT-large: 18x fewer parameters
- trained 1.7x faster
- regularization and generalization
- empirical evidence:
    - scale much better
    - SOTA on GLUE, RACE and SQUAD
    - RACE -> 44.1%(2017), 83.2% -> 89.4% (45.3% imporvement)
- RACE accuracy: 89.4%, GLUE benchmark: 89.4, F1 score SQUAD 2.0: 92.2


## References
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf)