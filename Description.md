## Case Study - Data Analysis with Human-Generated Text


In this document, we walk through some tips to help you with doing your own analysis on MIT EECS 
faculty data using stochastic variational inference on LDA.

1. Scraping your own dataset
2. Pre-processing the dataset
3. Implementing your own LDA code

**Implementing your own SVI-LDA code** 

Latent Dirichlet allocation (LDA) is a generative statistical model in natural language processing, and 
can be used to discover ‘topics’ in a large set of documents. This is first presented by David Blei, 
Andrew Ng, and Michael Jordan. 

The key idea is that if we see a ‘topic’ as a collection of certain 
words, we can look at each document as a collection of topics, the proportion of each topic depends 
on the proportion of words in the document that are associated with that topic. For example, the 
‘sports’ topic may consist of the words: tennis, football, gymnastics.
When given a set of documents, we can calculate the posterior distribution for the topics. In the 
original LDA paper, this is done using a coordinate descent algorithm for mean-field variational
inference, and later on researchers also used Gibbs Sampling and expectation propagation.
In this tutorial we will be looking only at Stochastic Variational Inference for LDA. SVI was first 
published in 2013 by Matt Hoffman, David Blei, Chong Wang, and John Paisley.

Traditional coordinate-descent variational inference requires each update to be carried out with all of the data, 
and these updates become inefficient when the dataset gets large as each update scales linearly 
with the size of the data. The key idea with SVI is to update global variational parameters more 
frequently.
Using local and global parameters, and given the dataset with a known number of datapoints, we 
could randomly take 1 data point at a time, update the local parameter, and project the change into 
the global parameters. Like traditional coordinate-descent variational inference, this is done until the 
result converges, i.e., the change in the global parameters is smaller than a certain value.
The implementation we will be talking about is a naive implementation of the algorithm described in 
the original paper

.
**Variable Notation**

Here we provide a brief overview of the input variables for LDA and SVI. Variables that can be set are 
the following: 

• λ: what we want in the end (the posterior distribution for the topics for each word

• vocab: this is the overall vocabulary we will have in the docs

• K: this is the number of topics we want to get in the end

• D: this is the total number of documents

• α: parameter for per-document topic distribution

• η: parameter for per-topic vocab distribution2017 © Massachusetts Institute of Technology

• τ: delay that down weights early iterations

• κ: forgetting rate, controls how quickly old information is forgotten; the larger the value, the 
slower it is.

• max:iterations: the number of maximum iterations the updates should go on for. We usually 
set a check such that if the difference in two consecutive values of λ is smaller than a certain 
value, we say the algorithm has converged. However, sometimes we could set this certain 
value too small, so we set a maximum iteration value to avoid updates running forever.

**LDA Generative Model**

We review the LDA generative model here. LDA assumes each document has K topics with different 
proportions. It models a corpus w of size D as follows:
    
• Draw distribution over vocabulary βk ~ Dirichlet(η) for topics k ∈ {1…K}

• For each document d ∈ {1…D} :
    
– Draw topic proportions θd ~ Dirichlet(α);

– For each word 𝑊𝑑 𝑛 in the document:
    
* Draw topic indicator 𝑍𝑑 𝑛~ Multinomial (θd)

* Draw word 𝑊𝑑 𝑛 ~ Multinomial (β𝑍𝑑𝑛)

Note that this model follows the ‘bag of words’ assumption, such that given the topic proportions, 
each word drawn is independent of any other words in the document.
