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
    Documents are considered files whose names end with .txt.
    Files in subdirectories are included.
    '''
    
    def __init__(self, directory):
        '''
        Constructor. By default, iterating over the corpus returns the tokens, 
        not their id's. Use `set_yield_ids` to change this behavior.
        cm.
        
        :param directory: the path to the directory containing the corpus
        '''
        # use unicode to make functions from os module return unicode objects
        # this is important to get the correct filenames
        self.directory = unicode(directory)
        self.yield_tokens = True
        self.length = sum(len(files) for _, _, files in os.walk(self.directory))
    
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
    
#     def __getitem__(self, index):
#         '''
#         Overload the [] operator. Return the i-th file in the observed directory.
#         Note that this is read only.
#         '''
#         return self.files[index]
    
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
        paragraphs = text.split('\n')
        sentences = []
        sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        
        for paragraph in paragraphs:
            # don't change to lower case yet in order not to mess with the
            # sentence splitter
            par_sentences = sent_tokenizer.tokenize(paragraph, 'pt')
            sentences.extend(par_sentences)
        
        return sentences
        
    def get_tokens_from_file(self, path): 
        '''
        Tokenize and preprocesses the given text.
        Preprocessing includes lower case and conversion of digits to 9.
        '''
        sentences = self.get_sentences_from_file(path)
        
        all_tokens = [stemmer.stemWords(utils.tokenize_sentence(sent, True))
                      for sent in sentences]
        
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
            if not filename.endswith('.txt'):
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
                 load_metadata=False, metadata_directory=None, stopwords=None):
        '''
        :param load_metadata: whether to load previously saved metadata
        :param metadata_directory: the directory where the metadata is stored.
            If None, defaults to the current directory.
        '''
        CorpusManager.__init__(self, corpus_directory)
        
        if stopwords is not None:
            self.stopwords = stopwords
        else:
            self.stopwords = set()
        
        file_acess = FileAccess(metadata_directory)
        if load_metadata:
            with open(file_acess.corpus_manager, 'rb') as f:
                data = cPickle.load(f)
            self.__dict__.update(data)
            logging.info('Loaded corpus metadata from {}'.format(file_acess.corpus_manager))
            logging.info('{} total sentences'.format(self.length))
        else:
            self.length = self._compute_length(self.directory)
            data = {'length': self.length}
            with open(file_acess.corpus_manager, 'wb') as f:
                cPickle.dump(data, f, -1)
            
            logging.info('Saved corpus metadata to {}'.format(file_acess.corpus_manager))
    
    def _compute_length(self, root_dir):
        '''
        Compute the total number of sentences in all files inside `root_dir`.
        '''
        num_sents = 0
        logging.info('Counting total number of sentences in directory {}'.format(root_dir))
        for root, _, files in os.walk(root_dir):
            for filename in files:
                if not filename.endswith('.txt'):
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
                if not filename.endswith('.txt'):
                    continue
                
                sentences = self.get_sentences_from_file(full_path)
                for sentence in sentences:
                    tokens = stemmer.stemWords(token
                                               for token in utils.tokenize_sentence(sentence, preprocess=True)
                                               if token not in self.stopwords)
                
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
    
    Only process .txt files. Any other extension is ignored.
    '''
    def __init__(self, directory, pre_tokenized=False):
        '''
        :param pre_tokenized: indicate that the corpus has already been tokenized;
            tokens should be separated by whitespace.
        '''
        self.directory = unicode(directory)
        self.yield_tokens = True
        self.pre_tokenized = pre_tokenized
        self._load_corpus()        
        
    def _load_corpus(self):
        '''
        Load the corpus to memory. Exactly repeated sentences are removed.
        '''
        # use an ordered dict as a set that mantains order
        corpus_sentences = OrderedDict()
        self.tokenized_cache = {}
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
                        tokens = tokenized_sent.split()
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
        tokens = stemmer.stemWords(utils.tokenize_sentence(sentence))
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
        
        self._file_num = 0
        self._sent_num = None
