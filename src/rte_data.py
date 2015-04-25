# -*- coding: utf-8 -*-

'''
Script para extrair pares (T, H) de notícias para inferência 
textual. A primeira linha da notícia deve ser seu título,
interpretado como H. A primeira sentença da segunda linha  
(a primeira do texto da matéria em si) é considerada T.
'''

from __future__ import unicode_literals

import argparse
import os
from xml.dom import minidom
from xml.etree import cElementTree as ET

import nltk

class Pair(object):
    '''
    Classe vazia apenas para armazenar uma estrutura (T, H)
    '''
    def __init__(self, t, h, **attribs):
        '''
        :param attribs: Extra attributes to be written to the XML file.
        '''
        self.t = t
        self.h = h
        self.t_attribs = {}
        self.h_attribs = {}
        self.attribs = attribs
    
    def set_t_attributes(self, **attribs):
        self.t_attribs = attribs
        
    def set_h_attributes(self, **attribs):
        self.h_attribs = attribs
    
    def __repr__(self):
        return u'RTE pair (T: {}, H: {})'.format(self.t, self.h).encode('utf-8')
    
    def __str__(self):
        return u'T: {}\nH: {}'.format(self.t, self.h).encode('utf-8')

def write_rte_file(filename, pairs, task='', entailment='UNKNOWN', **attribs):
    '''
    Write an XML file containing the given RTE pairs.
    
    :param pairs: list of Pair objects
    :parma task: the task attribute in the XML elements
    :param entailment: the entailment attribute. Should be either
        'YES', 'NO' or 'UNKNOWN'.
    '''
    root = ET.Element('entailment-corpus')
    for i, pair in enumerate(pairs, 1):
        xml_attribs = {'id':str(i), 'task':task, 'entailment':entailment}
        
        # add any other attributes supplied in the function call or the pair
        xml_attribs.update(attribs)
        xml_attribs.update(pair.attribs)
        
        xml_pair = ET.SubElement(root, 'pair', xml_attribs)
        xml_t = ET.SubElement(xml_pair, 't', pair.t_attribs)
        xml_h = ET.SubElement(xml_pair, 'h', pair.h_attribs)
        xml_t.text = pair.t.strip()
        xml_h.text = pair.h.strip()
    
#     tree = ET.ElementTree(root)
#     tree.write(filename, 'utf-8', True)
    
    # produz XML com formatação legível (pretty print)
    xml_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(xml_string)
    with open(filename, 'wb') as f:
        f.write(reparsed.toprettyxml('    ', '\n', 'utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Diretório com arquivo de entrada com notícias')
    parser.add_argument('output', help='Arquivo XML para ser salvo no formato RTE')
    args = parser.parse_args()
    
    items = os.listdir(args.input)
    pairs = []
    for item in items:
        # itera sobre arquivos
        full_path = os.path.join(args.input, item)
        if not os.path.isfile(full_path):
            continue
        
        with open(full_path, 'rb') as f:
            text = f.read().decode('utf-8')
        
        paragraphs = text.split('\n')
        title = ''
        while title == '':
            title = paragraphs.pop(0).strip()
        
        first_paragraph = ''
        while first_paragraph == '':
            first_paragraph = paragraphs.pop(0).strip()
        
        sentences = nltk.tokenize.sent_tokenize(first_paragraph, 'portuguese')
        t = sentences[0]
        h = title
        pair = Pair(t, h)
        pairs.append(pair)
    
    write_rte_file(args.output, pairs, task='', entailment='YES', 
                   length='short', origin='newswire')
    
    
