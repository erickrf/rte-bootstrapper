# -*- coding: utf-8 -*-

'''
Script to generate candidate pairs to the RTE task.
'''

import os
import logging
import argparse

from corpusmanager import InMemorySentenceCorpusManager
from rte_data import write_rte_file
from vectorspaceanalyzer import VectorSpaceAnalyzer

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
    parser.add_argument('--absolute-alpha', help='Minimum number of different tokens', type=int,
                        default=3, dest='absolute_alpha')
    parser.add_argument('--min-alpha', type=float, default=0.3, dest='min_alpha',
                        help='Minimum proportion of tokens exclusive to each sentence (default: 0.3)')
    parser.add_argument('--max-alpha', type=float, default=1, dest='max_alpha',
                        help='Maximum proportion of tokens exclusive to each sentence (default: 1)')
    parser.add_argument('-o', '--output', help='File to save the pairs', default='rte.xml')
    
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)

    vsa = VectorSpaceAnalyzer()
    vsa.load_data(args.vsm)
    pairs = []

    # iterate over the clusters
    for cluster in os.listdir(args.clusters):
        cluster_path = os.path.join(args.clusters, cluster)
        new_pairs = vsa.find_rte_candidates_in_cluster(cluster_path, 
                                                       minimum_score=args.min_score,
                                                       maximum_score=args.max_score,
                                                       num_pairs=args.cluster_pairs, 
                                                       pairs_per_sentence=1, 
                                                       min_alpha=args.min_alpha,
                                                       max_alpha=args.max_alpha,
                                                       absolute_min_alpha=args.absolute_alpha)
        pairs.extend(new_pairs)
        
        # write within loop so partial results are visible
        write_rte_file(args.output, pairs)
