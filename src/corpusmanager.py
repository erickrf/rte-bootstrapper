# -*- coding: utf-8 -*-


import os
import re

import nltk
import utils

class CorpusManager(object):
    '''
    Class to manage huge corpora. It iterates over the documents in a directory. 
    '''
    
    def __init__(self, directory, stopwords=None):
        '''
        Constructor. By default, iterating over the dictionary returns the tokens, 
        not their id's. Use `set_yield_ids` to change this behavior.
        
        :param filenames: the name of the files containing the corpus.
        '''
        self.directory = directory
        self.yield_tokens = True
        self.files = os.listdir(self.directory)
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)
    
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
        return len(self.files)
    
    def __getitem__(self, index):
        '''
        Overload the [] operator. Return the i-th file in the observed directory.
        Note that this is read only.
        '''
        return self.files[index]
    
    def get_text_from_document(self, number):
        '''
        Return the text content from the n-th document in the corpus.
        '''
        path = os.path.join(self.directory, self[number])
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8')
        
        return text        
    
    def get_sentences_from_document(self, number):
        '''
        Return a list of sentences contained in the document, without any preprocessing
        or tokenization.
        '''
        text = self.get_text_from_document(number)
        
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
        
    def get_tokens_from_document(self, number): 
        '''
        Tokenize and preprocesses the given text.
        Preprocessing includes lower case and conversion of digits to 9.
        '''
        sentences = self.get_sentences_from_document(number)
        
        all_tokens = [token
                      for sent in sentences
                      for token in utils.tokenize_sentence(sent)]
        
        return all_tokens
    
    def __iter__(self):
        '''
        Yield the text from a document inside the corpus directory.
        Stopwords are filtered out.
        '''
        for i in range(len(self.files)):
            tokens = self.get_tokens_from_document(i)
            
            if self.yield_tokens:
                yield tokens
            else:
                yield self.dictionary.doc2bow(tokens)

