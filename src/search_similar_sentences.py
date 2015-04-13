# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from pandas.core.config_init import doc

'''
Script to search for similar sentences, candidates to being RTE pairs.
Both positive pairs and negative ones can be found; human judgements are 
required in a post-processing phase. 

The negative pairs found by the script should share words and concepts, 
making them non-trivial to classify correctly. 
'''

import logging
import re
import os
import argparse
import gensim

from corpusmanager import CorpusManager

class VectorSpaceAnalyzer(object):
    '''
    Class to analyze documents according to vector spaces.
    It evaluates document similarity in search of RTE candidates.
    '''
    def __init__(self, corpus, dictionary=None, stopwords=None,
                 tfidf=None, lsi=None, index=None, num_topics=100):
        '''
        Constructor
        
        :param corpus: directory containing corpus files or
            MM file saved by gensim
        :param dictionary: file with previously saved dictionary
        :param stopwords: file with stopwords (one per line)
        :param tfidf: file with tf-idf model saved
        :param lsi: file with LSI model saved
        :param index: file with saved index
        :param num_topics: number of LSI topics
        '''
        if os.path.isfile(corpus):
            self.cm = gensim.corpora.MmCorpus(corpus)
            using_corpus_manager = False
        else:
            self.cm = CorpusManager(corpus)
            using_corpus_manager = True
        
        self.create_dictionary(dictionary, stopwords)
        
        if using_corpus_manager:
            self.cm.set_yield_ids(self.token_dict)
        
        self.create_tfidf_model(tfidf)
        self.create_lsi_model(lsi, num_topics)
        
        # create similarity index
        self.create_index(index)

    def create_dictionary(self, filename=None, stopwords_file=None, minimum_df=2):
        '''
        Try to load the dictionary if the given filename is not None.
        If it is, create from the corpus.
        
        :param corpus_manager: an CorpusManager object
        :param filename: name of the file containing the saved dictionary.
        :param stopwords_file: name of the file containing stopwords
        :param minimum_df: the minimum document frequency a token must have in 
            order to be included in the dictionary.
        '''
        if filename is not None:
            self.token_dict = gensim.corpora.Dictionary.load(filename)
            return
        
        # start it empty and fill it iteratively
        self.token_dict = gensim.corpora.Dictionary()
        
        logging.info('Creating token dictionary')
        for document in self.cm:
            self.token_dict.add_documents([document])
        
        if stopwords_file is not None:
            # load all stopwords from the given file
            with open(stopwords_file, 'rb') as f:
                text = f.read().decode('utf-8')
            stopwords = text.split('\n')
            
            # check which words appear in the dictionary and remove them
            stop_ids = [self.token_dict.token2id[stopword] 
                        for stopword in stopwords 
                        if stopword in self.token_dict.token2id]
            self.token_dict.filter_tokens(stop_ids)
        
        # remove punctuation
        punct_ids = [self.token_dict.token2id[token] 
                     for token in self.token_dict.token2id 
                     if re.match('\W+$', token)]
        
        # remove rare tokens
        rare_ids = [token_id 
                    for token_id, docfreq in self.token_dict.dfs.iteritems() 
                    if docfreq < minimum_df]
        
        self.token_dict.filter_tokens(punct_ids + rare_ids)
        
        # remove common tokens (appearing in more than 90% of the docs)
        self.token_dict.filter_extremes(no_above=0.9)
        
        # reassign id's, in case tokens were deleted
        self.token_dict.compactify()
        
        self.token_dict.save('token-dict.dat')
    
    def create_tfidf_model(self, filename=None):
        '''
        Create a vector space model from the given data.
        '''
        if filename is not None:
            self.tfidf = gensim.models.TfidfModel.load(filename)
            return
            
        self.tfidf = gensim.models.TfidfModel(self.cm)
        self.tfidf.save('tfidf.dat')    
        
    def create_lsi_model(self, filename, num_topics):
        '''
        Load the LSI model from a file, if given. If it is not, create from scratch.
        '''
        if filename is not None:
            self.lsi = gensim.models.LsiModel.load(filename)
            return
        
        self.lsi = gensim.models.LsiModel(self.tfidf[self.cm], 
                                          id2word=self.token_dict, 
                                          num_topics=num_topics)
        self.lsi.save('lsi.dat')
        
    
    def create_index(self, filename):
        '''
        Load the index if the given filename is not None. If it is, create it
        from the corpus.
        '''
        if filename is not None:
            self.index = gensim.similarities.Similarity.load(filename)
            return
        
        tf_idf_corpus = self.tfidf[self.cm]
        self.index = gensim.similarities.Similarity('shard', 
                                                    self.lsi[tf_idf_corpus],
                                                    self.lsi.num_topics)
        
        self.index.save('index.dat')
    
    def find_similar_documents(self, tokens, number=10):
        '''
        Find and return the most similar documents to the one represented
        by tokens.
        '''
        # create a bag of words from the document
        bow = self.token_dict.doc2bow(tokens)
        
        # create its tf-idf representation from the bag
        tfidf_repr = self.tfidf[bow]
        
        # and LSI representation from the tf-idf one
        lsi_repr = self.lsi[tfidf_repr]
        
        similarities = self.index[lsi_repr]
        
        # the similarities array contains the simliraty value for each document
        # we pick the indices in the order that would sort it
        indices = similarities.argsort()[-number:][::-1]
        
        # get the document names
        docs = [self.cm[idx]
                for idx in indices]
        
        return docs
    
    def find_rte_candidates(self, sentence):
        '''
        Find and return RTE candidates to be used with the given sentence.
        '''
        # exclude the first one because it is 
        similar_docs = self.find_similar_documents(sentence, 5)[1:]
        for doc in similar_docs:
            doc_content = self.cm.get_tokens_from_document(doc, True)
            
    
    def find_similar_sentences(self, target_sentence, sentences, number=10):
        '''
        Find and return similar sentences from the target one among all the
        given sentences.
        
        :param target_sentence: a list of tokens
        :param sentences: a list of lists of tokens (or something that
            can yield lists of tokens when iterated
        :param number: number of sentences to find
        '''
        sentences_tfidf = self.tfidf[sentences]
        sentences_lsi = self.lsi[sentences_tfidf]
        
        # create an index over the given sentences
        sentence_index = gensim.similarities.MatrixSimilarity(sentences_lsi)
        
        # now convert the target
        bow = self.token_dict.doc2bow(target_sentence)
        target_tfidf = self.tfidf[bow]
        target_lsi = self.lsi[target_tfidf]
        similarities = sentence_index[target_lsi]
        
        # get the indices of the most similar sentences
        indices = similarities.argsort()[-number:][::-1]
        
        results = [sentences[idx] for idx in indices]
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dictionary', help='Previously saved gensim dictionary')
    parser.add_argument('-s', '--stopwords', help='File containing stopwords')
    parser.add_argument('-t', '--topics', help='Number of LSA topics (default 100)', default=100)
    parser.add_argument('-i', '--index', help='Previously saved index')
    parser.add_argument('--tf-idf', help='TF-IDF model file', dest='tfidf')
    parser.add_argument('--lsi', help='LSI model file')
    parser.add_argument('corpus', help='Directory containing corpus files or MM file with corpus contents')
    parser.add_argument('query', help='Query for similar documents')
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    
    vsa = VectorSpaceAnalyzer(args.corpus, args.dictionary, args.stopwords, 
                              args.tfidf, args.lsi, args.index, args.topics)
    docs = vsa.find_similar_documents(args.query.decode('latin1').split(), 5)
    for doc in docs:
        print doc
    
    
    