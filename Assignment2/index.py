#!/usr/bin/python

import math

def is_alphabet(uchar):
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def check_word(word, st):
    split_noatation = ' '
    position = len(word)
    for i in range(st, len(word)):
        letter = word[i]
        if letter.isdigit():
            continue
        if not is_alphabet(letter):
            split_noatation = letter
            position = i
            break
    ret = word[st:].split(split_noatation)
    return ret, position

def pre_process(passage):
    candidates = passage.strip()[: -1].split()
    temp = []
    words = []
    for candidate in candidates:
        splited_words, pos = check_word(candidate, 0)
        # print(candidate, '->', splited_words)
        while pos < len(candidate) - 1:
            temp.append(splited_words[0])
            splited_words, pos = check_word(candidate, pos + 1)
            # print('   ', splited_words, pos)
        temp.append(splited_words[0])

    for word in temp:
        if len(word) < 4:
            continue
        if not is_alphabet(word[0]):
            continue
        if word[-1] == 's':
            if len(word) < 5:
                continue
            else:
                word = word[:-1]
        words.append(word)

    return words

class Document:
    def __init__(self, did, passages):
        self.did = did
        self.passages = passages
        self.frequency = {}
        for word in passages:
            if word in self.frequency.keys():
                self.frequency[word] += 1
            else:
                self.frequency[word] = 1

        self.frequency_rank = sorted(self.frequency.items(), key = lambda x: x[1], reverse = True)

        self.length = 0
        for word in self.frequency.keys():
            self.length += self.frequency[word] ** 2
        self.length = math.sqrt(self.length)

    def get_id(self):
        return self.did

    def get_frequency(self, keyword):
        return self.frequency[keyword]

    def get_length(self):
        return self.length

    def get_words(self):
        return self.frequency.keys()

    def get_top_n(self, n):
        return self.frequency_rank[: n]

class Query:
    def __init__(self, query):
        self.query = query.strip().split()
        self.frequency = {}
        for word in self.query:
            if word in self.frequency.keys():
                self.frequency[word] += 1
            else:
                self.frequency[word] = 1

        self.length = 0
        for word in self.frequency.keys():
            self.length += self.frequency[word] ** 2
        self.length = math.sqrt(self.length)

    def get_frequency(self, keyword):
        return self.frequency[keyword]

    def get_length(self):
        return self.length

    def get_words(self):
        return self.frequency.keys()

class InvertedFileIndex:
    def __init__(self, keywords, keywordindex):
        self.key_words = keywords
        self.positing_list = keywordindex

    def get_relevant(self, keyword):
        return self.key_words[keyword]

    def display(self, keyword):
        candidate_passages = self.key_words[keyword]
        out = keyword + ' ' * (10 - len(keyword))
        out += '-> | '
        for did in candidate_passages:
            out += 'D' + str(did) + ':'
            out += str(self.positing_list[did][keyword][0])
            for i in range(1, len(self.positing_list[did][keyword])):
                out += ',' + str(self.positing_list[did][keyword][i])
            out += ' | '
        print(out)

def load_data(input_path):
    key_words = {}
    positing_list = []
    documents = []
    input_collection = open(input_path, 'r')
    for line in input_collection:
        if len(line) < 2:
            continue
        words = pre_process(line)
        document = Document(len(documents), words)
        word_index = {}
        for i in range(len(words)):
            word = words[i]
            if word in word_index.keys():
                word_index[word].append(i)
            else:
                word_index[word] = [i]

            if word in key_words.keys():
                if key_words[word][-1] != len(documents):
                    key_words[word].append(len(documents))
            else:
                key_words[word] = [len(documents)]

        positing_list.append(word_index)
        documents.append(document)

    return key_words, positing_list, documents

def cosSim(document, query):
    similarity = 0
    for word in query.get_words():
        if word in document.get_words():
            similarity += query.get_frequency(word) * document.get_frequency(word)

    return similarity / (document.get_length() * query.get_length())

def do_query(query, documents, index):
    relevant = set()
    for word in query.get_words():
        candidate = index.get_relevant(word)
        for did in candidate:
            relevant.add(did)

    relevancy = {}
    for did in relevant:
        relevancy[did] = cosSim(documents[did], query)

    ret = sorted(relevancy.items(), key = lambda x: x[1], reverse = True)[: 3]
    return ret

def display_result(result, documents, index):
    for item in result:
        print('DID: %s' % item[0])
        top_n = documents[item[0]].get_top_n(5)
        for word, frequency in top_n:
            index.display(word)
        print('Number of unique keywords in document: %s' % len(documents[item[0]].get_words()))
        print('Magnitude of the document vector: %s' % 'Wait for implement')
        print('Similarity score: %s\n' % item[1])

def main():
    INPUT_PATH = './collection-100.txt'
    QUERY_PATH = './query-10.txt'

    key_words, positing_list, documents = load_data(INPUT_PATH)
    # for i, k in enumerate(key_words.keys()):
    #     print(i, k, key_words[k])

    index = InvertedFileIndex(key_words, positing_list)
    # index.display('bank')

    query = Query('bank')
    result = do_query(query, documents, index)
    display_result(result, documents, index)

if __name__ == '__main__':
    main()