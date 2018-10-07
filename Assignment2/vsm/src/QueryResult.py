
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
