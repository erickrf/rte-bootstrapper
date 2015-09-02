RTE Bootstrapper
================

This is a simple system to extract candidate RTE pairs from news clusters. 

Pairs are selected based on the similarity computed by vector space models. The purpose of this system is to extract both positive pairs (i.e., pairs where one sentence entails the other) and negative ones (pairs without any entailment relation); it is NOT expected to act as an unsupervised entailment detector. 

The point of using a VSM is to detect similar sentences (although the concept of similarity is somewhat subjective). By capturing pairs with high similarity, we are more likely to get positive ones, and also to avoid completely unrelated pairs as negative examples, which would be trivial to work with and not representative of a real world scenario.

Usage
-----

Requirements
~~~~~~~~~~~~

To use the RTE Bootstrapper, you need two corpora:

1) A very large corpus from which the vector space model will be generated. This corpus doesn't need to be clusterized. A directory in the file system should contain the corpus as collection of plain text files named with a ``.txt`` extension. The files can be directly under the corpus directory or inside subdirectories.

2) A corpus composed of clusters of related texts, from which the candidate pairs will be extracted. It should be a directory in the file system containing one subdirectory for each text cluster. The subdirectories should contain plain text files named with a ``.txt`` extension.

You will also need gensim_ and nltk_.

.. _gensim: https://radimrehurek.com/gensim/
.. _nltk: http://www.nltk.org/

Scripts
~~~~~~~

There are three scripts to be executed:

1) ``vectorspaceanalyzer.py`` creates the actual vector space model from a large corpus. **Note:** the corpus used with this script should **NOT** be the same with the clusters from which pairs will be selected.

2) ``create_index.py`` indexes all clusters (with respect to the VSM previously generated) for the subsequent pair extraction.

3) ``find_rte_candidates.py`` reads the indices from all cluster directories and extracts RTE candidate pairs from each one.

Detailed instructions for each script can be found by calling them on the command line with the ``-h`` flag. Some examples are shown below:

.. code:: bash

    python vectorspaceanalyzer.py /path/to/corpus stopwords.txt lda --dir /path/to/vsm/ -n 200

This will generate an LDA model with 200 topics from the corpus in the specified path, using the given stopwords file to remove words. The model will be saved in the path given by ``--dir``.