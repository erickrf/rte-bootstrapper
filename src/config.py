# -*- coding: utf-8 -*-

import os

class FileAccess(object):
    '''
    Class for storing file names used by the modules in this package.
    '''
    dictionary = 'token-dict.dat'
    stopwords = 'stopwords.txt'
    tfidf = 'tfidf.dat'
    lsi = 'lsi.dat'
    index = 'index.dat'
    lda = 'lda.dat'
    vsa_metadata = 'vsa-metadata.dat'
    rp = 'rp.dat'
    corpus_manager = 'corpus-manager.dat'
    hdp = 'hdp.dat'
    
    def __init__(self, directory=None):
        '''
        Configure the object to use the given directory.
        '''
        if directory is None:
            return
        
        # set all variables to include the directory
        # we do it this way in order to allow the variable written in the code
        # and thus accessible via code completion
        for attribute_name in vars(FileAccess):
            if attribute_name.startswith('__'):
                continue
            
            value = getattr(self, attribute_name)
            setattr(self, attribute_name, os.path.join(directory, value))
    