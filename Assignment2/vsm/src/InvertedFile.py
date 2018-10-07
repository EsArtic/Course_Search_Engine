
class InvertedFile(object):
    '''
        Inverted file index for the documents.

        Attrs:
            index: dictionary, map keywords to a list of document id.
    '''
    def __init__(self, word_file_map):
        self.__index = word_file_map

    def get_documents(self, term):
        if term in self.__index.keys():
            return self.__index[term]
        else:
            return None

    def exist(self, term):
        if term in self.__index.keys():
            return True
        else:
            return False
