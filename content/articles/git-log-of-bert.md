Title: A Commit History of BERT and its Forks
Date: 2020-05-09 17:18
Modified: 2020-05-09 17:18
Category: nlp
Slug: git-log-of-bert
Summary: What a commit history of version-controlled research papers could look like?
Status: published
Authors: Amit Chaudhary
Cover: /images/research-paper-as-forks.png

I recently came across an interesting thread on Twitter discussing a hypothetical scenario where research papers are published on GitHub and subsequent papers are diffs over the original paper. Information overload has been a real problem in ML with so many new papers coming every month.  
<div class="img-center">     
<blockquote class="twitter-tweet tw-align-center" data-lang="en" data-dnt="true"><p lang="en" dir="ltr">If you could represent a paper as a code diff, many papers could be compressed down to &lt;50 lines :) The diff would also be more intuitive to read and eval standardized.<br><br>Some ideas are so different that this wouldn’t apply, but I think it would work well for the majority. <a href="https://t.co/JoAcIK9Cm7">https://t.co/JoAcIK9Cm7</a></p>&mdash; Denny Britz (@dennybritz) <a href="https://twitter.com/dennybritz/status/1254006850388983808?ref_src=twsrc%5Etfw">April 25, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
</div>
<br>
This post is a fun experiment showcasing how the commit history could look like for the BERT paper and some of its subsequent variants.
![](/images/research-paper-as-forks.png){.img-center}
<br>

<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1810.04805">arXiv:1810.04805</a></p>   
<p>Author: Devlin et al.</p>  
<p>Date:   Thu Oct 11 00:50:01 2018 +0000</p>
<h2 style="padding-left: 1rem;">  
Initial Commit: BERT
</h2>  
<p style="color: #aa3131;">-Transformer Decoder</p>    
<p style="color: #7a942e;">+Masked Language Modeling</p>  
<p style="color: #7a942e;">+Next Sentence Prediction</p>  
<p style="color: #7a942e;">+WordPiece 30K</p>  
  </div>
</article>

<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1901.07291">arXiv:1901.07291</a></p>   
<p>Author: Lample et al.</p>  
<p>Date:   Sun Nov 10 10:46:37 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
Cross-lingual Language Model Pretraining
</h2>  
<p style="color: #7a942e;">+Translation Language Modeling(TLM)</p>  
<p style="color: #7a942e;">+Causal Language Modeling(CLM)</p>  
  </div>
</article>


<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1906.08237">arXiv:1906.08237</a></p>   
<p>Author: Yang et al.</p>  
<p>Date:   Wed Jun 19 17:35:48 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
XLNet: Generalized Autoregressive Pretraining for Language Understanding
</h2>  
<p style="color: #aa3131;">-Masked Language Modeling</p>    
<p style="color: #aa3131;">-BERT Transformer</p>    
<p style="color: #7a942e;">+Permutation Language Modeling</p>  
<p style="color: #7a942e;">+Transformer-XL</p>  
<p style="color: #7a942e;">+Two-stream self-attention</p>  
  </div>
</article>


<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1907.10529">arXiv:1907.10529</a></p>   
<p>Author: Joshi et al.</p>  
<p>Date:   Wed Jul 24 15:43:40 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
SpanBERT: Improving Pre-training by Representing and Predicting Spans
</h2>  
<p style="color: #aa3131;">-Random Token Masking</p>    
<p style="color: #aa3131;">-Next Sentence Prediction</p>    
<p style="color: #aa3131;">-Bi-sequence Training</p>    
<p style="color: #7a942e;">+Continuous Span Masking</p>  
<p style="color: #7a942e;">+Span-Boundary Objective(SBO)</p>  
<p style="color: #7a942e;">+Single-Sequence Training</p>  
  </div>
</article>


<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1907.11692">arXiv:1907.11692</a></p>   
<p>Author: Liu et al.</p>  
<p>Date:   Fri Jul 26 17:48:29 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
RoBERTa: A Robustly Optimized BERT Pretraining Approach
</h2>  
<p style="color: #aa3131;">-Next Sentence Prediction</p>    
<p style="color: #aa3131;">-Static Masking of Tokens</p>    
<p style="color: #7a942e;">+Dynamic Masking of Tokens</p>  
<p style="color: #7a942e;">+Byte Pair Encoding(BPE) 50K</p>  
<p style="color: #7a942e;">+Large batch size</p>  
<p style="color: #7a942e;">+CC-NEWS dataset</p>  
  </div>
</article>

<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1908.10084">arXiv:1908.10084</a></p>   
<p>Author: Reimers et al.</p>  
<p>Date:   Tue Aug 27 08:50:17 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
</h2>  
<p style="color: #7a942e;">+Siamese Network Structure</p>
<p style="color: #7a942e;">+Finetuning on SNLI and MNLI</p>
  </div>
</article>

<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1909.11942">arXiv:1909.11942</a></p>   
<p>Author: Lan et al.</p>  
<p>Date:   Thu Sep 26 07:06:13 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
</h2>  
<p style="color: #aa3131;">-Next Sentence Prediction</p>    
<p style="color: #7a942e;">+Sentence Order Prediction</p>  
<p style="color: #7a942e;">+Cross-layer Parameter Sharing</p>  
<p style="color: #7a942e;">+Factorized Embeddings</p>  
  </div>
</article>

<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1910.01108">arXiv:1910.01108</a></p>   
<p>Author: Sanh et al.</p>  
<p>Date:   Wed Oct 2 17:56:28 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
</h2>  
<p style="color: #aa3131;">-Next Sentence Prediction</p>    
<p style="color: #aa3131;">-Token-Type Embeddings</p>    
<p style="color: #aa3131;">-[CLS] pooling</p>    
<p style="color: #7a942e;">+Knowledge Distillation</p>  
<p style="color: #7a942e;">+Cosine Embedding Loss</p>  
<p style="color: #7a942e;">+Dynamic Masking</p>  
  </div>
</article>

<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1911.03894">arXiv:1911.03894</a></p>   
<p>Author: Martin et al.</p>  
<p>Date:   Sun Nov 10 10:46:37 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
CamemBERT: a Tasty French Language Model
</h2>  
<p style="color: #aa3131;">-BERT</p>    
<p style="color: #aa3131;">-English</p>    
<p style="color: #7a942e;">+ROBERTA</p>  
<p style="color: #7a942e;">+French OSCAR dataset(138GB)</p>  
<p style="color: #7a942e;">+Whole-word Masking(WWM)</p>  
<p style="color: #7a942e;">+SentencePiece Tokenizer</p>  
  </div>
</article>

<article class="message is-dark">
<div class="message-body">
<p style="color: #A57705;">commit <a href="https://arxiv.org/abs/1912.05372">arXiv:1912.05372</a></p>   
<p>Author: Le et al.</p>  
<p>Date:   Wed Dec 11 14:59:32 2019 +0000</p>
<h2 style="padding-left: 1rem;">  
FlauBERT: Unsupervised Language Model Pre-training for French
</h2>  
<p style="color: #aa3131;">-BERT</p>    
<p style="color: #aa3131;">-English</p>   
<p style="color: #7a942e;">+ROBERTA</p>  
<p style="color: #7a942e;">+fastBPE</p>  
<p style="color: #7a942e;">+Stochastic Depth</p>  
<p style="color: #7a942e;">+French dataset(71GB)</p>  
<p style="color: #7a942e;">+FLUE(French Language Understanding Evaluation) benchmark</p>  
  </div>
</article>