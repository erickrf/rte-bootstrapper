# -*- coding: utf-8 -*-

'''
Read one or more files and create a list of all sentences used in them.
Input files can be either plain text (one sentence per line) or the RTE XML
format.

This is useful to list all sentences previously added to XML files,
and then avoid using them when creating a new file.
'''

from xml.etree import cElementTree as ET
import argparse

def read_sentences(filename):
    '''
    Read the sentences in the file. This function delegates the call to either
    read_txt or read_xml.
    '''
    if filename.endswith('.txt'):
        return read_txt(filename)
    elif filename.endswith('.xml'):
        return read_xml(filename)
    else:
        raise ValueError('Invalid file extension')

def read_txt(filename):
    '''
    Read a text file with one sentence per line
    '''
    with open(filename, 'rb') as f:
        text = unicode(f.read(), 'utf-8')
        sentences = text.splitlines()
    
    return sentences

def read_xml(filename):
    '''
    Read all sentences in a XML RTE file.
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    sentences = []
    
    for pair in root:
        t = pair.find('t')
        s1 = t.text
        h = pair.find('h')
        s2 = h.text
        sentences.append(s1)
        sentences.append(s2)
    
    return sentences

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='*', help='Input files (XML or plain text)')
    parser.add_argument('output', help='Output file')
    args = parser.parse_args()
    
    sentences = set()
    for filename in args.input:
        file_sentences = read_sentences(filename)
        sentences.update(file_sentences)
    
    text = '\n'.join(sentences)
    
    with open(args.output, 'wb') as f:
        f.write(text.encode('utf-8'))
    