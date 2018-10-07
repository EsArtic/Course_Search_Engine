#!/usr/bin/python

import sys

from VectorSpace import VSM

# COLLECTION_PATH = '../data/'
# QUERY_PATH = '../query/'
# OUTPUT_PATH = '../output/'

def main():
    collection = 'collection-100.txt'
    queries = 'query-10.txt'
    output = None

    if len(sys.argv) > 1:
        collection = sys.argv[1]

    if len(sys.argv) > 2:
        queries = sys.argv[2]

    vsm_object = VSM(collection)
    # vsm_object.do_query(['is', 'a', 'bank'])
    vsm_object.batch_query(queries)

if __name__ == '__main__':
    main()
