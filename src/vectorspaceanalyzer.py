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

import rte_data
from config import FileAccess
import corpusmanager

class VectorSpaceAnalyzer(object):
    '''
    Class to analyze documents according to vector spaces.
    It evaluates document similarity in search of RTE candidates.
    '''
    def __init__(self):
        '''
        Constructor. Call `generate_model` or `load_data` to do something 
        useful with this class.
        '''
        self.ignored_docs = set()
    
    def generate_model(self, corpus, data_directory, method='lsi', load_dictionary=False, 
                       stopwords_file=None, num_topics=100, **corpus_manager_args):
        '''
        Generate a VSM from the given corpus and save it to the given directory.
        
        :param corpus: directory containing corpus text files
        :param data_directory: directory where models will be saved
        :param method: the method used to create the VSM
        :param stopwords: file with stopwords (one per line)
        :param num_topics: number of VSM topics (ignored if method is hdp)
        :param load_dictionary: load a previously saved dictionary
        :param corpus_manager_args: named arguments supplied to the corpus manager
            object created in this object.
        '''
        # load all stopwords from the given file
        if stopwords_file is not None:
            with open(stopwords_file, 'rb') as f:
                text = f.read().decode('utf-8')
            stopwords = set(text.split('\n'))
        
        self.cm = corpusmanager.SentenceCorpusManager(corpus, 
                                                      metadata_directory=data_directory,
                                                      stopwords=stopwords,
                                                      **corpus_manager_args)
        self.method = method
        self.num_topics = num_topics
        self.file_access = FileAccess(data_directory)
        
        if load_dictionary:
            self.token_dict = gensim.corpora.Dictionary.load(self.file_access.dictionary)
        else:
            self.create_dictionary()
        
        self.cm.set_yield_ids(self.token_dict)
        self.create_model()
        self.cm.save_configuration(self.file_access.corpus_manager)
        if self.method == 'hdp':
            # number of topics determined by the algorithm
            # (pretty hard to find, by the way)
            self.num_topics = self.hdp.m_lambda.shape[0]
        self.save_metadata()
        
    def save_metadata(self):
        '''
        Save metadata describing the VSA object.
        '''
        data = {'method': self.method, 
                'num_topics': self.num_topics}
        
        filename = self.file_access.vsa_metadata
        with open(filename, 'wb') as f:
            cPickle.dump(data, f, -1)
    
    def create_model(self):
        '''
        Create the VSM used by this object.
        '''
        if self.method == 'lsi':
            self.create_tfidf_model()
            self.create_lsi_model()
        elif self.method == 'lda':
            self.create_tfidf_model()
            self.create_lda_model()
        elif self.method == 'rp':
            self.create_rp_model()
        elif self.method == 'hdp':
            self.create_hdp_model()
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
            transformed_tfidf = self.tfidf[bag_of_words]
            return self.lda[transformed_tfidf]
        
        elif self.method == 'rp':
            return self.rp[bag_of_words]
        elif self.method == 'hdp':
            return self.hdp[bag_of_words]
        else:
            raise ValueError('Unknown VSM method: {}'.format(self.method))
    
    def create_dictionary(self):
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
        
        # remove punctuation
        punct_ids = [self.token_dict.token2id[token] 
                     for token in self.token_dict.token2id 
                     if re.match('\W+$', token)]
        
        self.token_dict.filter_tokens(punct_ids)
        
        # remove common and rare tokens (gensim by default filters out words 
        # appearing in less than 5 or more than 50% of the documents)
        self.token_dict.filter_extremes(no_below=20, keep_n=50000)
        
        # reassign id's, in case tokens were deleted
        self.token_dict.compactify()
        
        filename = self.file_access.dictionary
        self.token_dict.save(filename)
    
    def load_data(self, directory):
        '''
        Load the models from the given directory.
        '''
        self.file_access = FileAccess(directory)
        with open(self.file_access.vsa_metadata, 'rb') as f:
            metadata = cPickle.load(f)
        
        self.__dict__.update(metadata)
        self.token_dict = gensim.corpora.Dictionary.load(self.file_access.dictionary)
        
        if self.method == 'lsi':
            self.tfidf = gensim.models.TfidfModel.load(self.file_access.tfidf)
            self.lsi = gensim.models.LsiModel.load(self.file_access.lsi)
        elif self.method == 'lda':
            self.tfidf = gensim.models.TfidfModel.load(self.file_access.tfidf)
            self.lda = gensim.models.LdaModel.load(self.file_access.lda)
        elif self.method == 'rp':
            self.rp = gensim.models.RpModel.load(self.file_access.rp)
        elif self.method == 'hdp':
            self.hdp = gensim.models.HdpModel.load(self.file_access.hdp)
    
    # TODO: organize the following model creation functions avoiding repeated code
    # (I'm unwilling to use setattr and getattr though) 
    def create_tfidf_model(self):
        '''
        Create a TF-IDF vector space model from the given data.
        '''
        self.tfidf = gensim.models.TfidfModel(self.cm)
        filename = self.file_access.tfidf
        self.tfidf.save(filename) 
    
    def create_hdp_model(self):
        self.hdp = gensim.models.HdpModel(self.cm, id2word=self.token_dict)
        filename = self.file_access.hdp
        self.hdp.save(filename)
    
    def create_lsi_model(self):
        '''
        Create a LSI model from the corpus
        '''
        self.lsi = gensim.models.LsiModel(self.tfidf[self.cm], 
                                          id2word=self.token_dict, 
                                          num_topics=self.num_topics)
        filename = self.file_access.lsi
        self.lsi.save(filename)
    
    def create_rp_model(self):
        '''
        Create an RP model (Random Projections) 
        '''
        self.rp = gensim.models.RpModel(self.cm,
                                        id2word=self.token_dict,
                                        num_topics=self.num_topics)
        filename = self.file_access.rp
        self.rp.save(filename, [])
    
    def create_lda_model(self):
        '''
        Create a LDA model from the corpus
        '''
        self.lda = gensim.models.LdaModel(self.cm,
                                              id2word=self.token_dict,
                                              num_topics=self.num_topics)
        filename = self.file_access.lda
        self.lda.save(filename)
    
    def create_index(self):
        '''
        Create a similarity index to be used with the corpus.
        '''
        vsm_repr = self.transform(self.cm)
        self.index = gensim.similarities.Similarity('shard', 
                                                    self.lsi[vsm_repr],
                                                    self.num_topics)
        filename = self.file_access.index
        self.index.save(filename)
    
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
    
    def create_index_for_cluster(self, cluster_dir, pre_tokenized=False):
        '''
        Create a gensim index file for the cluster in the given directory.
        '''
        scm_config = self.file_access.corpus_manager
        scm = corpusmanager.InMemorySentenceCorpusManager(cluster_dir, pre_tokenized, 
                                                          configuration_file=scm_config)
        scm.set_yield_ids(self.token_dict)
        vsm_repr = self.transform(scm)
        index = gensim.similarities.MatrixSimilarity(vsm_repr, num_features=self.num_topics)
        
        index_filename = 'index-{}-{}.dat'.format(self.method, self.num_topics)
        path = os.path.join(cluster_dir, index_filename)
        index.save(path)
    
    def find_rte_candidates_in_cluster(self, corpus_dir, pre_tokenized=False, 
                                       min_score=0.8, num_pairs=0,
                                       absolute_min_alpha=3,
                                       min_alpha=0.2, max_alpha=1,
                                       max_score=0.99,
                                       min_t_size=5, min_h_size=5,
                                       max_t_size=0, max_h_size=0,
                                       filter_out_t=lambda _: False,
                                       filter_out_h=lambda _: False,
                                       avoid_sentences=None):
        '''
        Find and return RTE candidates within the given documents.
        
        Each sentence is compared to all others.
        
        :param corpus_dir: the directory containing text files to be analyzed
        :param min_score: threshold sentence similarity should be above in order
            to be considered RTE candidates
        :param max_score: threshold sentence similarity should be below in order
            to be considered RTE candidates
        :param num_pairs: number of pairs to be extracted; 0 means indefinite
        :param absolute_min_alpha: minimum number of tokens exclusive to each sentence
        :param min_alpha: minimum proportion of tokens in each sentence
            that can't appear in the other
        :param max_alpha: maximum proportion of tokens in each sentence
            that can't appear in the other
        :param max_t_size: maximum size (in tokens) of the first sentence in the pair
        :param max_h_size: maximum size (in tokens) of the second sentence in the pair
        :param filter_out_t: customized function to filter out T sentences 
            (should return True if the sentence should be discarded)
        :param filter_out_h: same as filter_out_t, but for H
        :param avoid_sentences: list of sentences that should be avoided
        '''
        scm_config = self.file_access.corpus_manager
        scm = corpusmanager.InMemorySentenceCorpusManager(corpus_dir, pre_tokenized,
                                                          configuration_file=scm_config)
        scm.set_yield_tokens()
        
        try:
            index_filename = 'index-{}-{}.dat'.format(self.method, self.num_topics)
            path = os.path.join(corpus_dir, index_filename)
            index = gensim.similarities.MatrixSimilarity.load(path)
        except:
            logging.warn('Index was not generated. If you intend to perform multiple experiments'\
                         'on this cluster, consider indexing it first with the create_index method.')
            vsm_repr = self.transform(scm)
            index = gensim.similarities.MatrixSimilarity(vsm_repr, num_features=self.num_topics)
        
        # sentences already used to create pairs are ignored afterwards, in order 
        # to allow more variability
        ignored_sents = set()
        candidate_pairs = []
        
        if avoid_sentences is not None:
            ignored_sents.update(avoid_sentences)
                
        for i, base_tokens in enumerate(scm):
            base_sent = scm[i]
            if filter_out_t(base_sent):
                # drop sentences without ending punctuation
                # this filters out titles and image subtitles
                continue
            
            if base_sent in ignored_sents:
                continue
            
            # this set contains the sentence tokens except for stopwords and rare words
            base_content_words = set(token
                                     for token in base_tokens
                                     if token in self.token_dict.token2id)
            
            base_length = scm.sentence_sizes[i]
            if base_length < min_t_size:
                # discard very short sentences
                continue
            
            if max_t_size > 0 and base_length > max_t_size:
                # discard long sentences (considering stop words)
                continue
            
            bow = self.token_dict.doc2bow(base_tokens)
            vsm_repr = self.transform(bow)
            similarities = index[vsm_repr]
            
            # get the indices of the sentences with highest similarity
            # [::-1] revereses the order
            similarity_args = similarities.argsort()[::-1]
            
            for arg in similarity_args:
                similarity = similarities[arg]
                if similarity < min_score:
                    # too dissimilar. since similarities are sorted, next ones will only be worse
                    break
                
                if similarity >= max_score:
                    # essentially the same sentence
                    continue
                
                other_sent = scm[arg]
                other_tokens = scm.get_tokenized_sentence(arg)
                other_length = scm.sentence_sizes[arg]
                if filter_out_h(other_sent):
                    continue
                if other_sent in ignored_sents:
                    continue
                
                if other_length < min_h_size:
                    continue
                
                if max_h_size > 0 and other_length > max_h_size:
                    # discard long sentences (considering stop words)
                    continue
                 
                other_sent_content_words = set(token
                                               for token in other_tokens
                                               if token in self.token_dict.token2id)
                
                # check the difference in the two ways
                diff1 = base_content_words - other_sent_content_words
                diff2 = other_sent_content_words - base_content_words
                if len(diff1) < absolute_min_alpha or len(diff2) < absolute_min_alpha:
                    continue
                 
                proportion1 = len(diff1) / float(len(base_content_words))
                proportion2 = len(diff2) / float(len(other_sent_content_words))
                if proportion1 < min_alpha or proportion2 < min_alpha:
                    continue
                
                if proportion1 > max_alpha or proportion2 > max_alpha:
                    continue
                
                pair = rte_data.Pair(base_sent, other_sent, similarity=str(similarity),
                                     alpha1=str(proportion1), alpha2=str(proportion2))
                pair.set_t_attributes(sentence=str(i))
                pair.set_h_attributes(sentence=str(arg))
                candidate_pairs.append(pair)
                
                if len(candidate_pairs) == num_pairs:
                    return candidate_pairs
                
                ignored_sents.add(base_sent)
                ignored_sents.add(other_sent)
                
                # avoid using more than one H for the same T
                break
        
        return candidate_pairs
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', help='Directory containing corpus files')
    parser.add_argument('stopwords', help='Stopword file (one word per line)')
    parser.add_argument('-n', dest='num_topics', help='Number of VSM topics (default 100)',
                        default=100, type=int)
    parser.add_argument('-q', help='Quiet mode; suppress logging', action='store_true',
                        dest='quiet')
    parser.add_argument('method', help='Method to generate the vector space',
                        choices=['lsi', 'lda', 'rp', 'hdp'])
    parser.add_argument('-s', help='Use stemmer', action='store_true', dest='stemmer')
    parser.add_argument('--dir', help='Set a directory to load and save models')
    parser.add_argument('--load-dict', help='Load previously saved dictionary file', 
                        action='store_true', dest='load_dictionary')
    parser.add_argument('--load-corpus-metadata', dest='load_corpus_metadata',
                        action='store_true', 
                        help='Load previously saved corpus metadata. Only used by the '\
                        'SentenceCorpusManager')
    parser.add_argument('--pre-tokenized', help='Using a pre-tokenized corpus. Tokens should '\
                        'be sepatated by whitespaces and already pre-processed.',
                        action='store_true', dest='pre_tokenized')
    args = parser.parse_args()
    
    if not args.quiet:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                            level=logging.INFO)
    
    vsa = VectorSpaceAnalyzer()
    vsa.generate_model(args.corpus_dir, args.dir, args.method, args.load_dictionary, 
                       args.stopwords, args.num_topics, load_metadata=args.load_corpus_metadata,
                       pre_tokenized=args.pre_tokenized, use_stemmer=args.stemmer)
    