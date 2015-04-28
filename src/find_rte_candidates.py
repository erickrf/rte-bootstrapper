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
    parser.add_argument('--data', help='Directory containing vector space models '\
                        '(default: current)', default='.')
    parser.add_argument('--min-score', help='Minimum sentence similarity score', type=float,
                        default=0.7, dest='min_score')
    parser.add_argument('--max-score', help='Maximum sentence similarity score', type=float,
                        default=0.99, dest='max_score')
    parser.add_argument('--cluster-pairs', help='Candidate pairs per cluster', type=int,
                        default=2)
    parser.add_argument('--min-diff', help='Minimum number of different tokens', type=int,
                        default=3, dest='min_diff')
    parser.add_argument('--min-proportion-diff', type=float, default=0.3, dest='min_proportion_diff',
                        help='Minimum proportion of tokens exclusive to each sentence')
    parser.add_argument('-o', '--output', help='File to save the pairs', default='rte.xml')
    
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)

    vsa = VectorSpaceAnalyzer()
    vsa.load_data(args.data)
    pairs = []

    # iterate over the clusters
    for cluster in os.listdir(args.clusters):
        cluster_path = os.path.join(args.clusters, cluster)
#         scm = InMemorySentenceCorpusManager(cluster_path)
        pairs.extend(vsa.find_rte_candidates_in_cluster(cluster_path, minimum_score=args.min_score,
                                                        maximum_score=args.max_score,
                                                        num_pairs=args.cluster_pairs, 
                                                        pairs_per_sentence=1, 
                                                        minimum_sentence_diff=args.min_diff,
                                                        minimum_proportion_diff=args.min_proportion_diff))

        write_rte_file(args.output, pairs)
