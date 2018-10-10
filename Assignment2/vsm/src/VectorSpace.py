import time

from Vector import Vector
from QueryResult import QueryResult
from DataManager import DataManager

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
        passage = passage.lower()
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
        print('DID: %d' % (result.get_id() + 1))
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
        print('----------------------------------------')
        start = time.time()

        query = Vector(query)
        for word in query.get_terms():
            query.set_weight(word, query.get_tf(word))

        query_result = self.__data_manager.get_query_result(query)
        for result in query_result:
            self.display_result(result)
            print('----------------------------------------')

        end = time.time()
        print('Spended Time: %.6fs\n' % (end - start))

    def batch_query(self, input_path):
        input_queries = open(input_path, 'r')
        num = 1
        for line in input_queries:
            line = line.strip()
            print('Query %d: %s' % (num, line))
            query = self.pre_process(line)
            if len(query) == 0:
                print('No keyword remained after preprocessing.')
            self.do_query(query)
            num += 1