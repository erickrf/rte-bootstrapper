# -*- coding: utf-8 -*-


import os
import re

import nlpnet

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
    
    def get_tokens_from_document(self, number, split_sentences=False):
        '''
        Return the tokens (processed by `get_tokens_from_text`) from the i-th
        document.
        
        :param split_sentences: if True, tokens are given in lists
            representing sentences
        '''
        path = os.path.join(self.directory, self[number])
        with open(path, 'rb') as f:
            text = f.read()        
        
        return self.get_tokens_from_text(text, split_sentences)
    
    def get_tokens_from_text(self, text, split_sentences=False): 
        '''
        Tokenize and preprocesses the given text.
        Preprocessing includes lower case and conversion of digits to 9.
        
        :param split_sentences: if True, tokens are given in lists
            representing sentences
        '''
        sentences = nlpnet.tokenize(text, 'pt')
        
        # make a single list with all tokens in lower case, and replace
        # all digits by 9
        if split_sentences:
            tokens = [[re.sub(r'\d', '9', token.lower()) for token in sent]
                      for sent in sentences]
        else:
            tokens = [re.sub(r'\d', '9', token.lower())
                      for sent in sentences
                      for token in sent]
        
        return tokens
    
    def __iter__(self):
        '''
        Yield the text from a document inside the corpus directory.
        Stopwords are filtered out.
        '''
        for filename in self.files:
            path = os.path.join(self.directory, filename)
            with open(path, 'rb') as f:
                text = f.read()
            
            tokens = self.get_tokens_from_text(text)
             
            if self.yield_tokens:
                yield tokens
            else:
                yield self.dictionary.doc2bow(tokens)

