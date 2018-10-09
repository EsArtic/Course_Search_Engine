
class Vector(object):
    '''
        Abstract data structure for document and query, initialize by a list
        containing the preprocessed passage words: ['Showers', 'continued',
        'throughout' ...].

        Attrs:
            did: int, if did = -1, this vector is representing a query.
            term_frequency: dictionary, map keywords to its frequency.
            term_index: dictionary, map keywords to a list of positions in
            the document.
            weights: dictionary, map keywords to its weights by the scheme
            of tf / max_tf * idf.
    '''
    def __init__(self, text, did = -1):
        self.__did = did
        self.__term_frequency = {}
        self.__term_index = {}
        self.__weights = {}
        for i in range(len(text)):
            word = text[i]
            if word in self.__term_frequency.keys():
                self.__term_frequency[word] += 1
                self.__term_index[word].append(i)
            else:
                self.__term_frequency[word] = 1
                self.__term_index[word] = [i]
                self.__weights[word] = 0.0

    def set_weight(self, term, value):
        self.__weights[term] = value

    def get_id(self):
        return self.__did

    def get_max_tf(self):
        return sorted(self.__term_frequency.items(), key = lambda x: x[1])[-1][1]

    def get_tf(self, term):
        if term in self.__term_frequency.keys():
            return self.__term_frequency[term]
        else:
            return 0.0

    def get_terms(self):
        return self.__term_frequency.keys()

    def get_weight(self, term):
        if term in self.__weights.keys():
            return self.__weights[term]
        else:
            return 0.0

    def get_weights(self):
        return self.__weights.values()

    def get_top_n_terms(self, n):
        n = min(n, len(self.__weights.keys()))
        return sorted(self.__weights.items(), key = lambda x: x[1], reverse = True)[: n]

    def display_term_index(self, term):
        print(' D%d:' % (self.__did + 1), end = '')
        print('%d' % self.__term_index[term][0], end = '')
        for i in range(1, len(self.__term_index[term])):
            print(',%d' % self.__term_index[term][i], end = '')
        print(' |', end = '')
