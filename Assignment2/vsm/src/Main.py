#!/usr/bin/python

import sys
import argparse

from VectorSpace import VSM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--collection', type = str, required = True,
                        help = 'Path of the documents collection file')
    parser.add_argument('-q', '--query', type = str, required = True,
                        help = 'Path of the queries collection file')
    args = parser.parse_args()
    COLLECTION_FOLDER = '../collection'
    QUERY_FOLDER = '../query'
    collections = '%s/%s' % (COLLECTION_FOLDER, args.collection)
    queries = '%s/%s' % (QUERY_FOLDER, args.query)

    vsm_object = VSM(collections)
    vsm_object.batch_query(queries)

if __name__ == '__main__':
    main()
