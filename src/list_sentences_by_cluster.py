# -*- coding: utf-8 -*-

'''
Read one or more files and create a json file with
all sentences by cluster.

This is useful to list all sentences previously added to XML files,
and then avoid using them when creating a new file.
'''

import json
from collections import defaultdict
from xml.etree import cElementTree as ET
import argparse

def process_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    contents = defaultdict(set)
    
    for pair in root:
        t = pair.find('t')
        s1 = t.text
        h = pair.find('h')
        s2 = h.text
        cluster = pair.get('cluster')
        contents[cluster].update([s1, s2])
    
    return contents

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='*', help='Input XML files')
    parser.add_argument('output', help='Output file')
    args = parser.parse_args()
    
    all_clusters = defaultdict(set)
    for filename in args.input:
        file_clusters = process_file(filename)
        
        for cluster in file_clusters:
            new_data = file_clusters[cluster]
            all_clusters[cluster].update(new_data)
    
    # change sets to list in order to be JSON serializable
    for cluster in all_clusters:
        cluster_set = all_clusters[cluster]
        all_clusters[cluster] = list(cluster_set)

    with open(args.output, 'wb') as f:
        json.dump(all_clusters, f, indent=4)
    