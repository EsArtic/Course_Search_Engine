#!/usr/bin/python

import math
import time
import numpy as np

# INPUT_PATH = '../data/'
INPUT_PATH = './'
OUTPUT_PATH = '../output/'

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
    '''
    def __init__(self, text, did = -1):
        self.__did = did
        self.__term_frequency = {}
        self.__term_index = {}
        for i in range(len(text)):
            word = text[i]
            if word in self.__term_frequency.keys():
                self.__term_frequency[word] += 1
                self.__term_index[word].append(i)
            else:
                self.__term_frequency[word] = 1
                self.__term_index[word] = [i]

    def get_id(self):
        return self.__did

    def get_max_tf(self):
        return sorted(self.__term_frequency.items(), key = lambda x: x[1])[-1][1]

    def get_tf(self, term):
        return self.__term_frequency[term]

    def get_terms(self):
        return self.__term_frequency.keys()

    def display_term_index(self, word):
        print(' D%d:' % self.__did, end = '')
        print('%d' % self.__term_index[word][0], end = '')
        for i in range(1, len(self.__term_index[word])):
            print(',%d' % self.__term_index[word][i], end = '')
        print(' |', end = '')

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

class VectorSpace(object):
    '''
        Abstract data structure for the vectorspace. Maintains the documents'
        weights vector into a big matrix, while the row indexes are document
        id and column indexes are term id in the dictionary.

        Attrs:
            vectorspace: np.array, the matrix holding documents' weights vectors.
    '''
    def __init__(self, documents, keywords, inverted_file):
        self.__vectorspace = np.zeros((len(documents), len(keywords)))
        for document in documents:
            did = document.get_id()
            for word in document.get_terms():
                tid = keywords[word]
                idf = math.log(len(documents) / len(inverted_file.get_documents(word)), 2)
                max_tf = document.get_max_tf()
                tf = document.get_tf(word)
                self.__vectorspace[did, tid] = tf / max_tf * idf

    def get_weights_vector(self, did):
        return self.__vectorspace[did, :]

    def get_weight(self, did, tid):
        return self.__vectorspace[did, tid]

class QueryResult(object):
    '''
        Abstract data structure for query result.
        Attrs:
            did: int, document id for the matched document.
            posting_list: list, containing the top five weighted keywords of the
            matched document, every items containing the keyword and the list of
            document id. E.g. ['even', [29, 59]].
            num_keywords: int, number of unique keywords in document.
            magnitude: float, magnitude of the document vector.
            sim_score: float, similarity score.
    '''
    def __init__(self, did, postinglist, numkeywords, magnitude, simscore):
        self.__did = did
        self.__posting_list = postinglist
        self.__num_keywords = numkeywords
        self.__magnitude = magnitude
        self.__sim_score = simscore

    def get_id(self):
        return self.__did

    def get_list(self):
        return self.__posting_list

    def get_num(self):
        return self.__num_keywords

    def get_magnitude(self):
        return self.__magnitude

    def get_sim_score(self):
        return self.__sim_score

class DataManager(object):
    '''
        Storage of all the data structure and maintains them.

        Attrs:
            documents: list, storing all the documents in the system.
            inverted_file: InvertedFile, the inverted file index for the documents.
            vspace: VectorSpace, the set of all the documents' weights vectors.
            dictionary: dictionary, map keywords to their term id, which are the
            indexes in the vector space.
    '''
    def __init__(self, word_file_map, documents):
        self.__documents = documents
        self.__inverted_file = InvertedFile(word_file_map)
        self.__dictionary = {}
        for i, term in enumerate(word_file_map.keys()):
            self.__dictionary[term] = i

        self.__vspace = VectorSpace(documents, self.__dictionary, self.__inverted_file)

    def magnitude(self, vector):
        return np.linalg.norm(vector, ord = 2)

    def similarity(self, v1, v2):
        sim = np.dot(v1, v2)
        sim /= self.magnitude(v1) * self.magnitude(v2)
        return sim

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
                print('\'%s\' has not been collected in the dictionary.' % word)

        return candidates

    def get_top_n_terms(self, did, n):
        document = self.__documents[did]
        rank_list = {}
        for word in document.get_terms():
            rank_list[word] = self.__vspace.get_weight(did, self.__dictionary[word])

        n = min(n, len(rank_list.keys()))
        return sorted(rank_list.items(), key = lambda x: x[1], reverse = True)[: n]

    def get_query_result(self, query):
        ret = []
        rank_list = {}
        candidates = self.get_documents_by_terms(query.get_terms())
        query_vector = np.zeros(len(self.__dictionary))

        for word in query.get_terms():
            if word in self.__dictionary.keys():
                query_vector[self.__dictionary[word]] = query.get_tf(word)

        for did in candidates:
            document = self.__documents[did]
            sim = self.similarity(self.__vspace.get_weights_vector(did), query_vector)
            rank_list[did] = sim

        result = sorted(rank_list.items(), key = lambda x: x[1], reverse = True)[:3]

        for did, sim in result:
            document = self.__documents[did]
            magnitude = self.magnitude(self.__vspace.get_weights_vector(did))
            num_terms = len(document.get_terms())
            words = self.get_top_n_terms(did, 5)
            postinglist = []
            for word, weight in words:
                dids = self.get_documents_by_term(word)
                postinglist.append([word, dids])
            ret.append(QueryResult(did, postinglist, num_terms, magnitude, sim))

        return ret

    def display_posting_list(self, word, dids):
        for did in dids:
            self.__documents[did].display_term_index(word)

class VSM(object):
    '''
        The controller of the system, interact with upper layers and perform operations.

        Attrs:
            data_manager: DataManager, storing all the data structure in the system.
    '''
    def __init__(self, input_path):
        word_file_map, documents = self.load_documents(input_path)
        self.__data_manager = DataManager(word_file_map, documents)

    def pre_process(self, passage):
        passage = passage.strip()
        for i in range(len(passage)):
            char = passage[i]
            if char.isalpha() or char.isdigit():
                continue
            if char == ' ':
                continue
            passage = passage[: i] + ' ' + passage[i + 1 :]

        candidates = passage.split()

        words = []
        for candidate in candidates:
            if len(candidate) < 4:
                continue
            if not candidate.isalpha():
                continue
            if candidate[-1] == 's':
                if len(candidate) < 5:
                    continue
                else:
                    candidate = candidate[: -1]
            words.append(candidate)

        return words

    def load_documents(self, input_path):
        word_file_map = {}
        documents = []

        input_collection = open(input_path, 'r')
        for line in input_collection:
            if len(line) < 2:
                continue

            words = self.pre_process(line)
            curr_id = len(documents)
            document = Vector(words, curr_id)
            documents.append(document)

            for word in words:
                if word not in word_file_map.keys():
                    word_file_map[word] = [curr_id]
                elif word_file_map[word][-1] != curr_id:
                    word_file_map[word].append(curr_id)

        return word_file_map, documents

    def display_result(self, result):
        print('DID: %d' % result.get_id())
        for word, dids in result.get_list():
            print('%-8s -> |' % word, end = '')
            self.__data_manager.display_posting_list(word, dids)
            print()
        print('Number of unique keywords in document: %s'
              % result.get_num())
        print('Magnitude of the document vector: %.2f'
              % result.get_magnitude())
        print('Similarity score: %.2f' % result.get_sim_score())

    def do_query(self, query):
        print('Query: %s' % query)
        print('----------------------------------------')
        start = time.time()

        query = Vector(query)

        query_result = self.__data_manager.get_query_result(query)
        for result in query_result:
            self.display_result(result)
            print('----------------------------------------')

        end = time.time()
        print('Spended Time: %.6fs\n' % (end - start))

    def batch_query(self, input_path):
        input_queries = open(input_path, 'r')
        for line in input_queries:
            query = self.pre_process(line)

            self.do_query(query)

def main():
    collection = 'collection-100.txt'
    queries = 'query-10.txt'

    vsm_object = VSM(INPUT_PATH + collection)
    # vsm_object.do_query(['is', 'a', 'bank'])
    vsm_object.batch_query(INPUT_PATH + queries)

if __name__ == '__main__':
    main()
