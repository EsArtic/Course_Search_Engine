#!/usr/bin/python

import math
import numpy as np

class Vector:
    '''
        Data structure for document and query
        when did = -1, the vector object is a query
    '''
    def __init__(self, words, did = -1):
        self.text = words
        self.did = did
        self.term_frequency = {}
        for word in self.text:
            if word in self.term_frequency.keys():
                self.term_frequency[word] += 1
            else:
                self.term_frequency[word] = 1

        self.max_tf = sorted(self.term_frequency.items(), key = lambda x: x[1])[-1][1]

    def get_id(self):
        return self.did

    def get_single_words(self):
        return self.term_frequency.keys()

    def get_tf(self, term):
        return self.term_frequency[term]

    def get_max_tf(self):
        return self.max_tf

class InvertedFile:
    '''
        Inverted file index for the preprocessed documents
    '''
    def __init__(self, word_file_map, word_indexes):
        self.index = word_file_map
        self.posting_list = word_indexes

    def get_documents(self, term):
        return self.index[term]

    def display(self, term):
        candidate_passages = self.index[term]
        out = term + ' ' * (10 - len(term))
        out += '-> | '
        for did in candidate_passages:
            out += 'D' + str(did) + ':'
            out += str(self.posting_list[did][term][0])
            for i in range(1, len(self.posting_list[did][term])):
                out += ',' + str(self.posting_list[did][term][i])
            out += ' | '
        print(out)

class VectorSpace:
    '''
        Containing the matrix representation for the whole vector space
    '''
    def __init__(self, documents, keywords):
        self.vectorspace = np.zeros((len(documents), len(keywords)))
        self.documents = documents
        self.dictionary = {}
        for i, word in enumerate(keywords):
            self.dictionary[word] = i

    def compute_df(self, inverted_file):
        n, m = self.vectorspace.shape
        self.df = {}
        self.idf = {}
        for word in self.dictionary.keys():
            curr_df = len(inverted_file.get_documents(word))
            curr_idf = math.log(n / curr_df, 2)
            self.df[word] = curr_df
            self.idf[word] = curr_idf

    def set_weights(self, inverted_file):
        self.compute_df(inverted_file)
        n, m = self.vectorspace.shape
        for document in self.documents:
            max_tf = document.get_max_tf()
            for word in document.get_single_words():
                tf = document.get_tf(word)
                self.vectorspace[document.get_id(), self.dictionary[word]] = (tf / max_tf) * self.idf[word]

    def compute_query_weight(self, query):
        n, m = self.vectorspace.shape
        query_weight = np.zeros((m, 1))
        max_tf = query.get_max_tf()
        for word in query.get_single_words():
            tf = query.get_tf(word)
            query_weight[self.dictionary[word]] = (tf / max_tf) * self.idf[word]

        return query_weight

    def get_weight(self, i):
        return self.vectorspace[i, :]

    def get_terms(self):
        return self.dictionary.keys()

    def get_document(self, did):
        return self.documents[did]

    def get_word_id(self, term):
        return self.dictionary[term]

def split_word(word, st):
    '''
        Split the words containing punctuation marks. E.g. Mexico's
    '''
    split_notation = ' '
    position = len(word)
    for i in range(st, len(word)):
        letter = word[i]
        if letter.isdigit() or letter.isalpha():
            continue
        split_notation = letter
        position = i
        break
    ret = word[st :].split(split_notation)
    return ret, position

def pre_process(passage):
    '''
        Preprocess the given passage -- one line in the original file
    '''
    candidates = passage.strip()[: -1].split()
    temp = []
    words = []

    for candidate in candidates:
        splited_words, pos = split_word(candidate, 0)
        while pos < len(candidate) - 1:
            temp.append(splited_words[0])
            splited_words, pos = split_word(candidate, pos + 1)
        temp.append(splited_words[0])

    for word in temp:
        if len(word) < 4:
            continue
        if not word.isalpha():
            continue
        if word[-1] == 's':
            if len(word) < 5:
                continue
            else:
                word = word[: -1]
        words.append(word)

    return words

def load_documents(input_path):
    '''
        Load documents into the memory by given path
    '''
    word_file_map = {}
    word_indexes = []
    documents = []

    input_collection = open(input_path, 'r')
    for line in input_collection:
        if len(line) < 2:
            continue
        words = pre_process(line)
        curr_did = len(documents)
        document = Vector(words, curr_did)
        documents.append(document)

        local_word_index = {}
        for i in range(len(words)):
            word = words[i]
            if word not in local_word_index.keys():
                local_word_index[word] = [i]
            else:
                local_word_index[word].append(i)

            if word not in word_file_map.keys():
                word_file_map[word] = [curr_did]
            elif word_file_map[word][-1] != curr_did:
                word_file_map[word].append(curr_did)

        word_indexes.append(local_word_index)

    return word_file_map, word_indexes, documents

def initialize(input_path):
    '''
        Initialize the data structures
    '''
    word_file_map, word_indexes, documents = load_documents(input_path)
    inverted_file = InvertedFile(word_file_map, word_indexes)
    vspace = VectorSpace(documents, word_file_map.keys())
    vspace.set_weights(inverted_file)

    return inverted_file, vspace

def L2_norm(vec):
    return np.linalg.norm(vec, ord=2)

# def my_norm(vec):
#     result = 0
#     for elem in vec:
#         result += elem ** 2
#     return math.sqrt(result)

def do_query(query, query_weight, inverted_file, vspace):
    documents = set()
    similarities = {}
    for word in query.get_single_words():
        candidate = inverted_file.get_documents(word)
        documents.update(candidate)

    length_query = L2_norm(query_weight)
    for did in documents:
        weights = vspace.get_weight(did)
        length_document = L2_norm(weights)
        similarity = np.dot(weights, query_weight)[0]
        similarity /= length_query * length_document
        similarities[did] = similarity

    ret = sorted(similarities.items(), key = lambda x: x[1], reverse = True)[: 3]
    return ret

def get_top_n(did, vspace):
    weights = vspace.get_weight(did)
    temp = {}
    relevant_words = vspace.get_document(did).get_single_words()
    for word in relevant_words:
        word_id = vspace.get_word_id(word)
        word_weight = weights[word_id]
        temp[word] = word_weight

    ret = sorted(temp.items(), key = lambda x: x[1], reverse = True)[: 5]
    return ret

def display_result(result, inverted_file, vspace):
    for did, similarity in result:
        print('DID: %s' % did)
        top_n = get_top_n(did, vspace)
        for word, weight in top_n:
            inverted_file.display(word)
        print('Number of unique keywords in document: %s' % len(vspace.get_document(did).get_single_words()))
        print('Magnitude of the document vector: %s' % L2_norm(vspace.get_weight(did)))
        print('Similarity score: %s\n' % similarity)

def main():
    INPUT_PATH = './collection-100.txt'
    QUERY_PATH = './query-10.txt'

    inverted_file, vspace = initialize(INPUT_PATH)
    weights = vspace.get_weight(0)
    # print(L2_norm(weights), my_norm(weights))
    # words = vspace.get_terms()
    # for word in words:
    #     print('%15s | %s' % (word, weights[vspace.get_word_id(word)]))

    query = Vector(['bank'])
    query_weight = vspace.compute_query_weight(query)
    # print(query_weight)
    result = do_query(query, query_weight, inverted_file, vspace)
    display_result(result, inverted_file, vspace)

if __name__ == '__main__':
    main()