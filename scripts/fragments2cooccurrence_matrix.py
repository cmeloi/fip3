#!/usr/bin/env python3
import argparse
import sys
import csv

from fip.profiles import CooccurrenceProfile


def field2features(field, delimiter):
    features = [feature.strip() for feature in field.split(delimiter) if feature]
    return features


def input2feature_lists(args):
    column_numbers = args.column_numbers
    delimiter = args.feature_delimiter
    with args.input as inputfile:
        reader = csv.reader(inputfile)
        if args.skip_header:
            next(reader)
        for row in reader:
            feature_set = set()
            for column_index in column_numbers:
                feature_set.update(field2features(row[column_index], delimiter))
            yield feature_set


def create_cooccurrence_matrix(args):
    mx = CooccurrenceProfile.from_feature_lists(input2feature_lists(args))
    with args.output as out:
        mx.to_csv(out)


def main():
    parser = argparse.ArgumentParser(description="Script to process input SMILES, one per line, into BRICS fragments.")
    parser.add_argument('-i', '--input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="Path to the input file. Default STDIN.")
    parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="Path to the output file. Default STDOUT.")
    parser.add_argument('-c', '--column_numbers', nargs='+', type=int, default=[0],
                        help="Column numbers within csv to process. Default 0, i.e. only the first one.")
    parser.add_argument('-d', '--feature_delimiter', nargs='?', type=str, default=' ',
                        help="Delimiter between the processed feature strings, to split on. Default space.")
    parser.add_argument('-s', '--skip_header', nargs='?', type=bool, default=False,
                        help="Skip the first line of the input (usually a header). Default false.")
    args = parser.parse_args()
    create_cooccurrence_matrix(args)


if __name__ == "__main__":
    main()
