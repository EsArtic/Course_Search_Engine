#!/usr/bin/python

import math
import time

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
            weights: dictionary, map keywords to its weights by the scheme
            of tf / max_tf * idf.
    '''
    def __init__(self, text, did = -1):
        self.did = did
        self.term_frequency = {}
        self.term_index = {}
        self.weights = {}
        for i in range(len(text)):
            word = text[i]
            if word in self.term_frequency.keys():
                self.term_frequency[word] += 1
                self.term_index[word].append(i)
            else:
                self.term_frequency[word] = 1
                self.term_index[word] = [i]
                self.weights[word] = 0.0

    def set_weight(self, word, value):
        self.weights[word] = value

    def get_id(self):
        return self.did

    def get_max_tf(self):
        return sorted(self.term_frequency.items(), key = lambda x: x[1])[-1][1]

    def get_tf(self, term):
        if term in self.term_frequency.keys():
            return self.term_frequency[term]
        else:
            return 0.0

    def get_terms(self):
        return self.term_frequency.keys()

    def get_weight(self, term):
        if term in self.weights.keys():
            return self.weights[term]
        else:
            return 0.0

    def get_weights(self):
        return self.weights.values()

    def get_top_n_terms(self, n):
        n = min(n, len(self.weights.keys()))
        return sorted(self.weights.items(), key = lambda x: x[1], reverse = True)[: n]

    def display_term_index(self, word):
        print(' D%d:' % self.did, end = '')
        print('%d' % self.term_index[word][0], end = '')
        for i in range(1, len(self.term_index[word])):
            print(',%d' % self.term_index[word][0], end = '')
        print(' |', end = '')

class InvertedFile(object):
    '''
        Inverted file index for the documents.

        Attrs:
            index: dictionary, map keywords to a list of document id.
    '''
    def __init__(self, word_file_map):
        self.index = word_file_map

    def get_documents(self, term):
        if term in self.index.keys():
            return self.index[term]
        else:
            return None

    def exist(self, term):
        if term in self.index.keys():
            return True
        else:
            return False

class QueryResult:
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
        self.did = did
        self.posting_list = postinglist
        self.num_keywords = numkeywords
        self.magnitude = magnitude
        self.sim_score = simscore

    def get_id(self):
        return self.did

    def get_list(self):
        return self.posting_list

    def get_num(self):
        return self.num_keywords

    def get_magnitude(self):
        return self.magnitude

    def get_sim_score(self):
        return self.sim_score

class DataManager(object):
    '''
        Storage of all the data structure and maintains them.

        Attrs:
            documents: list, storing all the documents in the system.
            inverted_file: InvertedFile, the inverted file index for the documents.
            dictionary: dictionary, map keywords to their term id, which are the
            indexes in the vector space.
    '''
    def __init__(self, word_file_map, documents):
        self.documents = documents
        self.inverted_file = InvertedFile(word_file_map)
        self.dictionary = {}
        for i, term in enumerate(word_file_map.keys()):
            self.dictionary[term] = i

        for document in self.documents:
            did = document.get_id()
            for word in document.get_terms():
                idf = math.log(len(documents) / len(word_file_map[word]), 2)
                max_tf = document.get_max_tf()
                tf = document.get_tf(word)
                weight = tf / max_tf * idf
                self.documents[did].set_weight(word, weight)

    def magnitude(self, vector):
        accumulate = 0
        for weight in vector.get_weights():
            accumulate += weight ** 2
        return math.sqrt(accumulate)

    def similarity(self, v1, v2):
        sim = 0.0
        for word in v2.get_terms():
            sim += v1.get_weight(word) * v2.get_weight(word)

        sim /= self.magnitude(v1) * self.magnitude(v2)
        return sim

    def get_query_result(self, query):
        ret = []
        rank_list = {}
        candidates = self.get_documents_by_terms(query.get_terms())
        for did in candidates:
            document = self.documents[did]
            sim = self.similarity(document, query)
            rank_list[did] = sim

        result = sorted(rank_list.items(), key = lambda x: x[1], reverse = True)[:3]

        for did, sim in result:
            document = self.documents[did]
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
        return self.inverted_file.get_documents(word)

    def get_documents_by_terms(self, words):
        candidates = set()
        have_illegal_words = False
        illegal_words = []
        for word in words:
            if self.inverted_file.exist(word):
                candidates.update(self.inverted_file.get_documents(word))
            else:
                have_illegal_words = True
                illegal_words.append(word)

        if have_illegal_words:
            for word in illegal_words:
                print('\'%s\' has not been collected in the dictionary.' % word)

        return candidates

    def display_posting_list(self, word, dids):
        for did in dids:
            self.documents[did].display_term_index(word)

class VSM(object):
    '''
        The controller of the system, interact with upper layers and perform operations.

        Attrs:
            data_manager: DataManager, storing all the data structure in the system.
    '''
    def __init__(self, input_path):
        word_file_map, documents = self.load_documents(input_path)
        self.data_manager = DataManager(word_file_map, documents)

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
            self.data_manager.display_posting_list(word, dids)
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
        for word in query.get_terms():
            query.set_weight(word, query.get_tf(word))

        query_result = self.data_manager.get_query_result(query)
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
    # vsm_object.batch_query(INPUT_PATH + queries)

if __name__ == '__main__':
    main()
