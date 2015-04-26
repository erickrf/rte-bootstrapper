# -*- coding: utf-8 -*-

from __future__ import unicode_literals

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
import cPickle
import gensim

import utils
import rte_data
from config import FileNames
import corpusmanager

class VectorSpaceAnalyzer(object):
    '''
    Class to analyze documents according to vector spaces.
    It evaluates document similarity in search of RTE candidates.
    '''
    def __init__(self, corpus, load_data=False, method='lsi', stopwords=None, num_topics=100,
                 load_dictionary=False):
        '''
        Constructor
        
        :param corpus: directory containing corpus files or None if you only 
            want to use saved models (LSI, TF-IDF).
        :param load_data: load data from previously saved files. The 
            other parameters are ignored if this is True.
        :param method: the method used to create the VSM
        :param stopwords: file with stopwords (one per line)
        :param num_topics: number of LSI topics (ignored if load_data is True)
        :param load_dictionary: load a previously saved dictionary, but generate
            a new VSM (only makes sense if load_data is False)
        '''
        if corpus is not None:
            self.cm = corpusmanager.SentenceCorpusManager(corpus)
        
        if load_data:
            self._load_data()
        else:
            self.method = method
            self.num_topics = num_topics
            
            if load_dictionary:
                self.token_dict = gensim.corpora.Dictionary.load(FileNames.dictionary)
            else:
                self.create_dictionary(stopwords)
            
            self.cm.set_yield_ids(self.token_dict)
            self.create_model()
            self.save_metadata()
        
        self.ignored_docs = set()
    
    def save_metadata(self):
        '''
        Save metadata describing the VSA object.
        '''
        data = {'method': self.method, 
                'num_topics': self.num_topics}
        with open(FileNames.vsa_metadata, 'wb') as f:
            cPickle.dump(data, f, -1)
    
    def create_model(self):
        '''
        Create the VSM used by this object.
        '''
        if self.method == 'lsi':
            self.create_tfidf_model()
            self.create_lsi_model()
        elif self.method == 'lda':
            # doesn't need TF-IDF
            self.create_lda_model()
        elif self.method == 'rp':
            self.create_rp_model()
        else:
            raise ValueError('Unknown VSM method: {}'.format(self.method))
    
    def transform(self, bag_of_words):
        '''
        Transform the given bag of words in a vector space representation
        according to the method used by this object.
        '''
        if self.method == 'lsi':
            transformed_tfidf = self.tfidf[bag_of_words]
            return self.lsi[transformed_tfidf]
        elif self.method == 'lda':
            return self.lda[bag_of_words]
        elif self.method == 'rp':
            return self.rp[bag_of_words]
        else:
            raise ValueError('Unknown VSM method: {}'.format(self.method))
    
    def create_dictionary(self, stopwords_file=None, minimum_df=2):
        '''
        Try to load the dictionary if the given filename is not None.
        If it is, create from the corpus.
        
        :param filename: name of the file containing the saved dictionary.
        :param stopwords_file: name of the file containing stopwords
        :param minimum_df: the minimum document frequency a token must have in 
            order to be included in the dictionary.
        '''
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
        
        self.token_dict.save(FileNames.dictionary)
    
    def _load_data(self):
        '''
        Load the models from default locations.
        '''
        with open(FileNames.vsa_metadata, 'rb') as f:
            metadata = cPickle.load(f)
        self.__dict__.update(metadata)
        
        self.token_dict = gensim.corpora.Dictionary.load(FileNames.dictionary)
        
        if self.method == 'lsi':
            self.tfidf = gensim.models.TfidfModel.load(FileNames.tfidf)
            self.lsi = gensim.models.LsiModel.load(FileNames.lsi)
        elif self.method == 'lda':
            self.lda = gensim.models.LdaModel.load(FileNames.lda)
        elif self.method == 'rp':
            self.rp = gensim.models.RpModel.load(FileNames.rp)
    
    # TODO: organize the following model creation functions avoiding repeated code
    # (I'm unwilling to use setattr and getattr though) 
    def create_tfidf_model(self):
        '''
        Create a TF-IDF vector space model from the given data.
        '''
        self.tfidf = gensim.models.TfidfModel(self.cm)
        self.tfidf.save(FileNames.tfidf)    
        
    def create_lsi_model(self):
        '''
        Create a LSI model from the corpus
        '''
        self.lsi = gensim.models.LsiModel(self.tfidf[self.cm], 
                                          id2word=self.token_dict, 
                                          num_topics=self.num_topics)
        self.lsi.save(FileNames.lsi)
    
    def create_rp_model(self):
        '''
        Create an RP model (Random Projections) 
        '''
        self.rp = gensim.models.RpModel(self.cm,
                                        id2word=self.token_dict,
                                        num_topics=self.num_topics)
        self.rp.save(FileNames.rp)
    
    def create_lda_model(self):
        '''
        Create a LDA model from the corpus
        '''
        self.lda = gensim.models.LdaModel(self.cm,
                                          id2word=self.token_dict,
                                          num_topics=self.num_topics)
        self.lda.save(FileNames.lda)
    
    def create_index(self):
        '''
        Create a similarity index to be used with the corpus.
        '''
        vsm_repr = self.transform(self.cm)
        self.index = gensim.similarities.Similarity('shard', 
                                                    self.lsi[vsm_repr],
                                                    self.num_topics)
        
        self.index.save(FileNames.index)
    
    def find_similar_documents(self, tokens, number=10, return_scores=True):
        '''
        Find and return the ids of the most similar documents to the one represented
        by tokens.
        
        :param return_scores: if True, return instead a tuple (ids, similarities)
        '''
        # create a bag of words from the document
        bow = self.token_dict.doc2bow(tokens)
        vsm_repr = self.transform(bow)
        similarities = self.index[vsm_repr]
        
        # the similarities array contains the simliraty value for each document
        # we pick the indices in the order that would sort it
        indices = similarities.argsort()
        
        # [::-1] reverses the order, so we have the greatest values first
        indices = indices[::-1]
        
        top_indices = []
        # exclude the first one because it is the compared document itself
        for index in indices[1:]:
            if index in self.ignored_docs:
                continue
            
            top_indices.append(index)
            
            if len(top_indices) == number:
                break
        
        if return_scores:
            return (top_indices, similarities[top_indices])
        else:
            return top_indices
    
    def find_rte_candidates_in_cluster(self, scm=None, sent_threshold=0.8, num_pairs=0, 
                                       pairs_per_sentence=1,
                                       minimum_sentence_diff=3,
                                       minimum_proportion_diff=0.1):
        '''
        Find and return RTE candidates within the given documents.
        
        Each sentence is compared to all others.
        
        :param scm: a SentenceCorpusManager object, containing a text cluster
        :param sent_threshold: threshold sentence similarity should be above in order
            to be considered RTE candidates
        :param num_pairs: number of pairs to be extracted; 0 means indefinite
        :param pairs_per_sentence: number of pairs a sentence can be part of
        :param minimum_sentence_diff: minimum number of tokens exclusive to each sentence
        :param minimum_proportion_diff: minimum proportion of tokens in each sentence
            that can't appear in the other
        '''
        scm.set_yield_ids(self.token_dict)
        
        vsm_repr = self.transform(scm)
        index = gensim.similarities.MatrixSimilarity(vsm_repr, num_features=self.num_topics)
        
        # sentences already used to create pairs are ignored afterwards, in order 
        # to allow more variability
        ignored_sents = set()
        candidate_pairs = []
        
        for i, sent in enumerate(scm):
            if len(sent) < 5:
                # discard very short sentences
                # stopwords are pruned before this check
                continue
            
            base_sent = scm[i]
            base_tokens = utils.tokenize_sentence(base_sent)
            base_token_set = set(base_tokens)
            
            vsm_repr = self.transform(sent)
            similarities = index[vsm_repr]
            
            # get the indices of the sentences with highest similarity
            # [::-1] revereses the order
            similarity_args = similarities.argsort()[::-1]
            
            # counter to limit the number of pairs per sentence 
            sentence_count = 0
            for arg in similarity_args:
                similarity = similarities[arg]
                if similarity < sent_threshold:
                    # too dissimilar. since similarities are sorted, next ones will only be worse
                    break
                
                if similarity >= 0.99 or arg in ignored_sents:
                    # essentially the same sentence, or already used
                    continue
                
                other_sent = scm[arg]
                other_tokens = utils.tokenize_sentence(other_sent)
                
                if len(other_tokens) < 5:
                    continue
                
                other_tokens_set = set(other_tokens)
                # check the difference in the two ways
                diff1 = base_token_set - other_tokens_set
                diff2 = other_tokens_set - base_token_set
                if len(diff1) < minimum_sentence_diff or len(diff2) < minimum_sentence_diff:
                    continue
                
                proportion1 = len(diff1) / float(len(base_token_set))
                proportion2 = len(diff2) / float(len(other_tokens_set))
                if proportion1 < minimum_proportion_diff or proportion2 < minimum_proportion_diff:
                    continue
                
                pair = rte_data.Pair(base_sent, other_sent, similarity=str(similarity))
                pair.set_t_attributes(sentence=str(i))
                pair.set_h_attributes(sentence=str(arg))
                candidate_pairs.append(pair)
                
                if len(candidate_pairs) == num_pairs:
                    return candidate_pairs
                
                sentence_count += 1
                if sentence_count >= pairs_per_sentence:
                    ignored_sents.add(i)
                    break
        
        return candidate_pairs
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', help='Directory containing corpus files')
    parser.add_argument('stopwords', help='Stopword file (one word per line)')
    parser.add_argument('-n', dest='num_topics', help='Number of VSM topics (default 100)',
                        default=100)
    parser.add_argument('-q', help='Quiet mode; suppress logging', action='store_true',
                        dest='quiet')
    parser.add_argument('method', help='Method to generate the vector space',
                        choices=['lsi', 'lda', 'rp'])
    parser.add_argument('--load-dict', help='Load previously saved dictionary file', 
                        action='store_true', dest='load_dictionary')
    parser.add_argument('--dir', help='Set a directory to load and save models')
    args = parser.parse_args()
    
    if not args.quiet:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                            level=logging.INFO)
    
    if args.dir is not None:
        initial_dir = os.getcwd()
        os.chdir(args.dir)
    
    vsa = VectorSpaceAnalyzer(args.corpus_dir, False, args.method, args.stopwords, 
                              args.num_topics, args.load_dictionary)
    
    if args.dir is not None:
        os.chdir(initial_dir)
    