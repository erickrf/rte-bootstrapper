# -*- coding: utf-8 -*-

'''
Utility functions.
'''

import re

from nltk.tokenize.regexp import RegexpTokenizer

def detokenize(tokens):
    '''
    Create a string from the given tokens, using whitespace where needed.
    '''
    s = ' '.join(tokens)
    s = re.sub(' ([.,;:?!()])', r'\1', s)
    return s

def tokenize_sentence(text, preprocess=True):
    '''
    Tokenize the given sentence and applies preprocessing if requested 
    (conversion to lower case and digit substitution).
    '''
    if preprocess:
        text = re.sub(r'\d', '9', text.lower())
    
    tokenizer_regexp = ur'''(?ux)
    ([^\W\d_]\.)+|                # one letter abbreviations, e.g. E.U.A.
    \d{1,3}(\.\d{3})*(,\d+)|      # numbers in format 999.999.999,99999
    \d{1,3}(,\d{3})*(\.\d+)|      # numbers in format 999,999,999.99999
    \d+:\d+|                      # time and proportions
    \d+([-\\/]\d+)*|              # dates. 12/03/2012 12-03-2012
    [DSds][Rr][Aa]?\.|            # common abbreviations such as dr., sr., sra., dra.
    [Mm]\.?[Ss][Cc]\.?|           # M.Sc. with or without capitalization and dots
    [Pp][Hh]\.?[Dd]\.?|           # Same for Ph.D.
    [^\W\d_]{1,2}\$|              # currency
    (?:(?<=\s)|^)[\#@]\w*[A-Za-z_]+\w*|  # Hashtags and twitter user names
    -[^\W\d_]+|                   # clitic pronouns with leading hyphen
    \w+([-']\w+)*|                # words with hyphens or apostrophes, e.g. não-verbal, McDonald's
    -+|                           # any sequence of dashes
    \.{3,}|                       # ellipsis or sequences of dots
    \S                            # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)
    
    return tokenizer.tokenize(text)
