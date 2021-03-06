# -*- coding: utf-8 -*-

'''
Script to create a VSM similarity index to the sentences
of all files contained in a directory. 
'''

import argparse
import logging
import os

from vectorspaceanalyzer import VectorSpaceAnalyzer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', help='Directory with clusters. Files in each one will be indexed.')
    parser.add_argument('vsa_dir', help='Directory containing saved Vector Space Analyzer')
    parser.add_argument('--pre-tokenized', help='Signal that the corpus has already been tokenized.',
                        action='store_true', dest='pre_tokenized')
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    vsa = VectorSpaceAnalyzer()
    vsa.load_data(args.vsa_dir)
    
    for item in os.listdir(args.corpus_dir):
        path = os.path.join(args.corpus_dir, item)
        if os.path.isdir(path):
            vsa.create_index_for_cluster(path, args.pre_tokenized)
