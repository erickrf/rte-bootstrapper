# -*- coding: utf-8 -*-

'''
Script to generate candidate pairs to the RTE task.
'''

import os
import logging
import json
import argparse

from vectorspaceanalyzer import VectorSpaceAnalyzer
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clusters', help='Directory containing news clusters')
    parser.add_argument('--vsm', help='Directory containing vector space models '\
                        '(default: current)', default='.')
    parser.add_argument('--min-score', help='Minimum sentence similarity score', type=float,
                        default=0.7, dest='min_score')
    parser.add_argument('--max-score', help='Maximum sentence similarity score', type=float,
                        default=0.99, dest='max_score')
    parser.add_argument('--cluster-pairs', help='Candidate pairs per cluster', type=int,
                        default=2)
    parser.add_argument('--avoid', help='A JSON file listing sentences per cluster that should be avoided. '\
                        'It can be created with the script list_sentences_by_cluster')
    parser.add_argument('--absolute-alpha', help='Minimum number of different tokens', type=int,
                        default=3, dest='absolute_alpha')
    parser.add_argument('--min-alpha', type=float, default=0.3, dest='min_alpha',
                        help='Minimum proportion of tokens exclusive to each sentence (default: 0.3)')
    parser.add_argument('--max-alpha', type=float, default=1, dest='max_alpha',
                        help='Maximum proportion of tokens exclusive to each sentence (default: 1)')
    parser.add_argument('--max-t-size', type=int, default=0, dest='max_t_size',
                        help='Maximum T size (first component in each pair)')
    parser.add_argument('--max-h-size', type=int, default=0, dest='max_h_size',
                        help='Maximum H size (second component in each pair)')
    parser.add_argument('--filter-prefixes', default=None,
                        help='Text file containing in each line a "prefix". Sentences starting with any\
                        of the prefixes are filtered out.')
    parser.add_argument('--pre-tokenized', action='store_true', dest='pre_tokenized',
                        help='Signal that the corpus has already been tokenized')
    parser.add_argument('-o', '--output', help='File to save the pairs', default='rte.xml')
    
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)

    vsa = VectorSpaceAnalyzer()
    vsa.load_data(args.vsm)
    
    prefixes = utils.read_lines(args.filter_prefixes)
    filter_ = utils.generate_filter(True, prefixes)
    
    writer = utils.XmlWriter(vsm=vsa.method)
    
    if args.avoid is not None:
        with open('avoid-sentences.txt', 'rb') as f:
            avoid_data = json.load(f)
    else:
        avoid_data = {}
    
    # iterate over the clusters
    for cluster in os.listdir(args.clusters):
        cluster_path = os.path.join(args.clusters, cluster)
        avoid_sentences = avoid_data.get(cluster)
        
        new_pairs = vsa.find_rte_candidates_in_cluster(cluster_path,
                                                       pre_tokenized=args.pre_tokenized,
                                                       min_score=args.min_score,
                                                       max_score=args.max_score,
                                                       num_pairs=args.cluster_pairs,
                                                       min_alpha=args.min_alpha,
                                                       max_alpha=args.max_alpha,
                                                       absolute_min_alpha=args.absolute_alpha,
                                                       min_t_size=7,
                                                       min_h_size=7,
                                                       max_t_size=args.max_t_size,
                                                       max_h_size=args.max_h_size,
                                                       filter_out_h=filter_,
                                                       filter_out_t=filter_,
                                                       avoid_sentences=avoid_sentences)
        
        writer.add_pairs(new_pairs, cluster)
            
    # pretty print 
    writer.write_file(args.output, True)
    