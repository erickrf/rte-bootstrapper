# -*- coding: utf-8 -*-

'''
Script to recursively tokenize all text files (.txt extension) in all
subdirectories of a directory in the file system. 

It will
1) rewrite existing files so they have one sentence per line
2) write a new file with the extension .token as a fully tokenized version
3) optionally, stem tokens

Files already tokenized are ignored.
'''

import argparse
import os
import logging
import nltk
import Stemmer

import utils

def recursive_run(directory, only_lines, only_tokens, stem):
    '''
    Recursively tokenizes files in a directory. It will call itself on 
    subdirectories.
    '''
    logger = logging.getLogger(__name__)
    
    logger.info('Entering directory %s' % directory)
    dir_contents = os.listdir(unicode(directory))
    files = 0
    
    for item in dir_contents:
        full_path = os.path.join(directory, item)        
        if os.path.isdir(full_path):
            recursive_run(full_path, only_lines, only_tokens, stem)
        
        if not item.endswith('.txt'):
            # only consider .txt files
            continue
        
        tokenized_path = full_path.replace('.txt', '.token')
        if os.path.isfile(tokenized_path):
            # there is already a tokenized file
            logger.info('Already tokenized, skipping: %s' % item)
            continue
        
        with open(full_path, 'rb') as f:
            text = unicode(f.read(), 'utf-8')
        
        paragraphs = text.split('\n')
        sentences = []
        sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        
        for paragraph in paragraphs:
            # don't change to lower case yet in order not to mess with the
            # sentence splitter
            par_sentences = sent_tokenizer.tokenize(paragraph, 'pt')
            sentences.extend(par_sentences)
        
        if not only_tokens:
            text = '\n'.join(sentences)
            with open(full_path, 'wb') as f:
                f.write(text.encode('utf-8'))
        
        if not only_lines:
            with open(tokenized_path, 'wb') as f:
                for sentence in sentences:
                    tokens = utils.tokenize_sentence(sentence, True)
                    if stem:
                        tokens = stemmer.stemWords(tokens)
                    line = '%s\n' % ' '.join(tokens)
                    f.write(line.encode('utf-8'))
        
        files += 1
    
    logger.info('Tokenized %d files' % files)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root_dir', help='Corpus root directory')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l', help='Only split existing files into one sentence per line '\
                       '(do not create tokenized files)', action='store_true', dest='only_lines')
    group.add_argument('-t', help='Only create tokenized files (do not split files into lines)',
                       action='store_true', dest='only_tokens')
    parser.add_argument('-s', help='Stem tokens', action='store_true', dest='stem')
    parser.add_argument('-v', action='store_true', help='Verbose',
                        dest='verbose')
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARN
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    
    if args.stem:
        stemmer = Stemmer.Stemmer('portuguese')
    
    logger.info('Starting to run')
    recursive_run(args.root_dir, args.only_lines, args.only_tokens, args.stem)
        