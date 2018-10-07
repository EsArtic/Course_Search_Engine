import math

from InvertedFile import InvertedFile
from QueryResult import QueryResult

class DataManager(object):
    '''
        Storage of all the data structure and maintains them.

        Attrs:
            documents: list, storing all the documents in the system.
            inverted_file: InvertedFile, the inverted file index for the documents.
    '''
    def __init__(self, word_file_map, documents):
        self.__documents = documents
        self.__inverted_file = InvertedFile(word_file_map)

        for document in self.__documents:
            did = document.get_id()
            for word in document.get_terms():
                idf = math.log(len(documents) / len(word_file_map[word]), 2)
                max_tf = document.get_max_tf()
                tf = document.get_tf(word)
                weight = tf / max_tf * idf
                self.__documents[did].set_weight(word, weight)

    def magnitude(self, vector):
        accumulate = 0
        for weight in vector.get_weights():
            accumulate += weight ** 2
        return math.sqrt(accumulate)

    def similarity(self, vec1, vec2):
        sim = 0.0
        for word in vec2.get_terms():
            sim += vec1.get_weight(word) * vec2.get_weight(word)

        sim /= self.magnitude(vec1) * self.magnitude(vec2)
        return sim

    def get_query_result(self, query):
        ret = []
        rank_list = {}
        candidates = self.get_documents_by_terms(query.get_terms())
        for did in candidates:
            document = self.__documents[did]
            sim = self.similarity(document, query)
            rank_list[did] = sim

        result = sorted(rank_list.items(), key = lambda x: x[1], reverse = True)[:3]

        for did, sim in result:
            document = self.__documents[did]
            magnitude = self.magnitude(document)
            num_terms = len(document.get_terms())
            words = document.get_top_n_terms(5)
            postinglist = []
            for word, weight in words:
                dids = self.get_documents_by_term(word)
                postinglist.append([word, dids])
            ret.append(QueryResult(did, postinglist, num_terms, magnitude, sim))

        return ret

    def get_documents_by_term(self, word):
        return self.__inverted_file.get_documents(word)

    def get_documents_by_terms(self, words):
        candidates = set()
        have_illegal_words = False
        illegal_words = []
        for word in words:
            if self.__inverted_file.exist(word):
                candidates.update(self.__inverted_file.get_documents(word))
            else:
                have_illegal_words = True
                illegal_words.append(word)

        if have_illegal_words:
            for word in illegal_words:
                print('\'%s\' has not been collected in the vocabulary.' % word)

        return candidates

    def display_posting_list(self, word, dids):
        for did in dids:
            self.__documents[did].display_term_index(word)
