# -*- coding: utf-8 -*-

'''
Script to create a VSM similarity index to the sentences
of all files contained in a directory. 
'''

import argparse
import logging

from vectorspaceanalyzer import VectorSpaceAnalyzer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', help='Directory with files where index will be created')
    parser.add_argument('vsa_dir', help='Directory containing saved Vector Space Analyzer')
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    vsa = VectorSpaceAnalyzer()
    vsa.load_data(args.vsa_dir)
    vsa.create_index_for_cluster(args.corpus_dir)
