# -*- coding: utf-8 -*-

'''
Utility functions.
'''

import re

import nlpnet

def detokenize(tokens):
    '''
    Create a string from the given tokens, using whitespace where needed.
    '''
    s = ' '.join(tokens)
    s = re.sub(' ([.,;:?!()])', r'\1', s)
    return s

def tokenize_sentence(text, preprocess=True):
    '''
    Tokenize the given sentence (already split into sentences) and applies 
    preprocessing if requested (conversion to lower case and digit substitution).
    '''
    if preprocess:
        text = re.sub(r'\d', '9', text.lower())
    
    return nlpnet.utils.tokenize_sentence_pt(text)
    