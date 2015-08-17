RTE Bootstrapper
================

This is a simple system to extract candidate RTE pairs from news clusters. 

Pairs are selected based on the similarity computed by vector space models. The purpose of this system is to extract both positive pairs (i.e., pairs where one sentence entails the other) and negative ones (pairs without any entailment relation); it is NOT expected to act as an unsupervised entailment detector. 

The point of using a VSM is to detect similar sentences (although the concept of similarity is somewhat subjective). By capturing pairs with high similarity, we are more likely nor only to get positive ones, but also to avoid completely unrelated pairs as negative examples, which would be trivial to work with and not representative of a real world scenario.

Usage
-----

To use the RTE Bootstrapper, you need a directory in the file system containing one subdirectory for each text cluster. The subdirectories should contain plain text files named with a `.txt` extension.

You will also need gensim_.

.. _gensim: https://radimrehurek.com/gensim/

