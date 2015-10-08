# -*- coding: utf-8 -*-


import os
import logging
import cPickle
from collections import OrderedDict
import nltk
import Stemmer

import utils
from config import FileAccess

stemmer = Stemmer.Stemmer('portuguese')

class CorpusManager(object):
    '''
    Class to manage huge corpora. It iterates over the documents in a directory.
    Documents are considered files whose names end with .txt (if not pre-tokenized)
    or .token (if pre-tokenized).
    Files in subdirectories are included.
    '''
    
    def __init__(self, directory, stopwords=None, pre_tokenized=False, use_stemmer=False):
        '''
        Constructor. By default, iterating over the corpus returns the tokens, 
        not their id's. Use `set_yield_ids` to change this behavior.
        cm.
        
        :param directory: the path to the directory containing the corpus
        :param stopwords: list or set of stopwords. If use_stemmer is True, stopwords
            are checked before applying stemmer.
        :param pre_tokenized: signal that the corpus is already tokenized; tokens
            are separated by white spaces
        :param use_stemmer: use a stemmer in tokens
        '''
        # use unicode to make functions from os module return unicode objects
        # this is important to get the correct filenames
        self.directory = unicode(directory)
        self.pre_tokenized = pre_tokenized
        self.file_extension = '.token' if pre_tokenized else '.txt'
        self.yield_tokens = True
        self.use_stemmer = use_stemmer
        self.length = sum(len(files) for _, _, files in os.walk(self.directory))
        self._set_stopwords(stopwords)
    
    def _set_stopwords(self, stopwords):
        '''
        Set the stopwords attribute to a dictionary containing the supplied ones 
        (which may be empty).
        '''
        if stopwords is not None:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set()
        
    def load_configuration(self, filename):
        '''
        Load the configuration file specific to the Corpus Manager, such as whether
        a stemmer is used.
        '''
        if not os.path.isfile(filename):
            logging.warning('Could not find configuration file %s' % filename)
            logging.warning('Assuming no stemmer is used')
            self.use_stemmer = False
            return
        
        with open(filename, 'rb') as f:
            data = cPickle.load(f)
        self.use_stemmer = data['use_stemmer']
        self.stopwords = data['stopwords']
        
    def save_configuration(self, filename):
        '''
        Save the configuration of the Corpus Manager
        '''
        data = {'use_stemmer': self.use_stemmer,
                'stopwords': self.stopwords}
        
        with open(filename, 'wb') as f:
            cPickle.dump(data, f, -1)
    
    def set_yield_tokens(self):
        '''
        Call this function in order to set the corpus manager to yield lists
        of tokens (instead of their id's).
        '''
        self.yield_tokens = True
    
    def set_yield_ids(self, dictionary):
        '''
        Call this function in order to set the corpus manager to yield the token
        id's (instead of the tokens themselves).
        '''
        self.yield_tokens = False
        self.dictionary = dictionary
    
    def __len__(self):
        '''
        Return the number of documents this corpus manager deals with.
        '''
        return self.length
    
    def get_text_from_file(self, path):
        '''
        Return the text content from the given path
        '''
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8')
        
        return text        
    
    def get_sentences_from_file(self, path):
        '''
        Return a list of sentences contained in the document, without any preprocessing
        or tokenization.
        '''
        text = self.get_text_from_file(path)
        
        # we assume that lines contain whole paragraphs. In this case, we can split
        # on line breaks, because no sentence will have a line break within it.
        # also, it helps to properly separate titles without a full stop
        lines = text.split('\n')
        
        # if the corpus is pre-tokenized, each line is a full sentence
        if self.pre_tokenized:
            return lines
        
        sentences = []
        sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        
        for paragraph in lines:
            # don't change to lower case yet in order not to mess with the
            # sentence splitter
            par_sentences = sent_tokenizer.tokenize(paragraph, 'pt')
            sentences.extend(par_sentences)
        
        return sentences
        
    def get_tokens_from_file(self, path): 
        '''
        Tokenize and preprocesses the given text.
        Preprocessing includes stemming, lower case and conversion of digits to 9.
        
        If the corpus was pre-tokenized, no pre-processing is done.
        '''
        sentences = self.get_sentences_from_file(path)
        all_tokens = []
        
        for sent in sentences:
            if self.pre_tokenized:
                tokens = [token for token in sent.split()
                          if token not in self.stopwords]
            else:
                tokens = [token for token in utils.tokenize_sentence(sent, True)
                          if token not in self.stopwords]
            
            if self.use_stemmer:
                tokens = stemmer.stemWords(tokens)
            
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def _iterate_on_dir(self, path):
        '''
        Internal helper recursive function.
        '''
        # sort the list because os.listdir returns files in arbitrary order,
        # and we want the result to be deterministic (in order to use the same
        # index through multiple runs in the same corpus)
        file_list = sorted(os.listdir(path))
        for filename in file_list:
            if not filename.endswith(self.file_extension):
                continue
            
            full_path = os.path.join(path, filename)
            if os.path.isdir(full_path):
                for item in self._iterate_on_dir(full_path):
                    yield item
            else:
                # this is a file
                tokens = self.get_tokens_from_file(full_path)
                
                if self.yield_tokens:
                    yield tokens
                else:
                    yield self.dictionary.doc2bow(tokens)

    def __iter__(self):
        '''
        Yield the text from a document inside the corpus directory.
        '''
        for item in self._iterate_on_dir(self.directory):
            yield item
                

class SentenceCorpusManager(CorpusManager):
    '''
    This class provides one sentence at a time. Documents are split into sentences
    on demand, but an initial run is needed in order to compute total corpus size.
    '''
    def __init__(self, corpus_directory,
                 load_metadata=False, metadata_directory=None, 
                 stopwords=None, pre_tokenized=False, use_stemmer=False):
        '''
        :param load_metadata: whether to load previously saved metadata
        :param metadata_directory: the directory where the metadata is stored.
            If None, defaults to the current directory.
        '''
        CorpusManager.__init__(self, corpus_directory, stopwords, pre_tokenized, use_stemmer)
        
        self.file_access = FileAccess(metadata_directory)
        if load_metadata:
            with open(self.file_access.corpus_metadata, 'rb') as f:
                data = cPickle.load(f)
            self.__dict__.update(data)
            logging.info('Loaded corpus metadata from {}'.format(self.file_access.corpus_metadata))
            logging.info('{} total sentences'.format(self.length))
        else:
            self.length = self._compute_length(self.directory)
            data = {'length': self.length}
            with open(self.file_access.corpus_metadata, 'wb') as f:
                cPickle.dump(data, f, -1)
            
            logging.info('Saved corpus metadata to {}'.format(self.file_access.corpus_metadata))
    
    def _compute_length(self, root_dir):
        '''
        Compute the total number of sentences in all files inside `root_dir`.
        '''
        num_sents = 0
        logging.info('Counting total number of sentences in directory {}'.format(root_dir))
        
        for root, _, files in os.walk(root_dir):
            for filename in files:
                if not filename.endswith(self.file_extension):
                    continue
                
                path = os.path.join(root, filename)
                num_sents += len(self.get_sentences_from_file(path))
        
        logging.info('Found {} sentences'.format(num_sents))
        return num_sents
    
    def __len__(self):
        return self.length
    
    def _iterate_on_dir(self, path):
        '''
        Internal helper recursive function.
        '''
        # sorted file list like in the parent class
        file_list = sorted(os.listdir(path))
        for filename in file_list:
            full_path = os.path.join(path, filename)
            if os.path.isdir(full_path):
                for item in self._iterate_on_dir(full_path):
                    yield item
            else:
                # this is a file
                if not filename.endswith(self.file_extension):
                    continue
                
                sentences = self.get_sentences_from_file(full_path)
                for sentence in sentences:
                    if self.pre_tokenized:
                        tokens = [token for token in sentence.split()
                                  if token not in self.stopwords]
                    else:
                        tokens = [token
                                  for token in utils.tokenize_sentence(sentence, preprocess=True)
                                  if token not in self.stopwords]
                    
                    if self.use_stemmer:
                        tokens = stemmer.stemWords(tokens)
                
                    if self.yield_tokens:
                        yield tokens
                    else:
                        yield self.dictionary.doc2bow(tokens)

class InMemorySentenceCorpusManager(CorpusManager):
    '''
    This class manages corpus access providing one sentence at a time.
    It must be used in a directory WITHOUT subdirectories.
    
    This class stores all corpus content in memory, so it should only 
    be used with small corpora.
    
    If not pre-tokenized, only process .txt files. Any other extension is ignored.
    If pre-tokenized, process .token files instead. 
    '''
    def __init__(self, directory, pre_tokenized=False, configuration_file=None,
                 use_stemmer=False, stopwords=None):
        '''
        :param pre_tokenized: indicate that the corpus has already been tokenized;
            tokens should be separated by whitespace.
        :param configuration_file: file saved by another Corpus Manager containing
            configuration data such as whether to use a stemmer and stopwords list.
            If given, the next two parameters are ignored.
        '''
        self.directory = unicode(directory)
        self.yield_tokens = True
        
        if configuration_file is None:
            self.use_stemmer = use_stemmer
            self._set_stopwords(stopwords)
        else:
            self.load_configuration(configuration_file)
        
        self.pre_tokenized = pre_tokenized
        self._load_corpus()
        
    def _load_corpus(self):
        '''
        Load the corpus to memory. Exactly repeated sentences are removed.
        '''
        # use an ordered dict as a set that mantains order
        corpus_sentences = OrderedDict()
        self.tokenized_cache = {}
        self.sentence_sizes = {}
        tokenized_sent_counter = 0
        
        # sort file names, like in the parent class iterator
        file_list = sorted(os.listdir(self.directory))
        for filename in file_list:
            
            if not filename.endswith('.txt'):
                continue
            
            path = os.path.join(self.directory, filename)
            
            if self.pre_tokenized:
                # read both the original and tokenized texts
                file_text = self.get_text_from_file(path)
                file_sentences = file_text.split('\n')
                
                path_tokenized = path.replace('.txt', '.token')
                tokenized_text = self.get_text_from_file(path_tokenized)
                tokenized_sentences = tokenized_text.split('\n')
                iter_tokenized = iter(tokenized_sentences)
            else:
                file_sentences = self.get_sentences_from_file(path)
            
            
            for sent in file_sentences:
                
                if self.pre_tokenized:
                    tokenized_sent = iter_tokenized.next()
                    if sent not in corpus_sentences:
                        # only include non-repeated sentences
                        tokens = tokenized_sent.split()
                        
                        # save the sentence length because after removing stopwords
                        # we'll have no way of knowing the original length
                        self.sentence_sizes[tokenized_sent_counter] = len(tokens)
                        
                        tokens = [token for token in tokens 
                                  if token not in self.stopwords]
                        
                        if self.use_stemmer:
                            tokens = stemmer.stemWords(tokens)
                        
                        self.tokenized_cache[tokenized_sent_counter] = tokens
                        tokenized_sent_counter += 1 
                
                # the value None is irrelevant here, we only need an ordered set 
                corpus_sentences[sent] = None
            
        self.sentences = corpus_sentences.keys()
        
    def get_tokenized_sentence(self, index):
        '''
        Return the sentence in the position indicated by the index properly tokenized.
        A cache is used to store sentences from the corpus previously tokenized.
        '''
        if index in self.tokenized_cache:
            return self.tokenized_cache[index]
        
        sentence = self[index]
        tokens = utils.tokenize_sentence(sentence)
        self.sentence_sizes[index] = len(tokens)
        tokens = [token for token in tokens
                  if token not in self.stopwords]
        
        if self.use_stemmer:
            tokens = stemmer.stemWords(tokens)
        
        self.tokenized_cache[index] = tokens
        
        return tokens
    
    def __getitem__(self, index):
        return self.sentences[index]
    
    def __len__(self):
        return len(self.sentences)
    
    def __iter__(self):
        '''
        Yield sentences.
        '''
        for i, _ in enumerate(self.sentences):
            
            tokens = self.get_tokenized_sentence(i)
            if self.yield_tokens:
                yield tokens
            else:
                yield self.dictionary.doc2bow(tokens)
        