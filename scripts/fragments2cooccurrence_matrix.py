#!/usr/bin/env python3
import argparse
import sys
import re

from fip.profiles import CooccurrenceProfile


def line2features(line, delimiter_regex):
    features = re.split(delimiter_regex, line.strip())
    for feature in features:
        if feature:
            yield feature.strip()


def input2feature_lists(input_lines, delimiters):
    delimiter_regex = re.compile('|'.join([delimiter.replace('|', '\|') for delimiter in delimiters]))
    for line in input_lines:
        line = line.strip()
        if line:
            yield line2features(line, delimiter_regex)


def create_cooccurrence_matrix(args):
    with args.output as out, args.input as inputfile:
        if args.skip_header:
            next(inputfile)
        mx = CooccurrenceProfile.from_feature_lists(input2feature_lists(inputfile, args.feature_delimiters))
        mx.to_csv(out)


def main():
    parser = argparse.ArgumentParser(description="Script to process input SMILES, one per line, into BRICS fragments.")
    parser.add_argument('-i', '--input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="Path to the input file. Default STDIN.")
    parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="Path to the output file. Default STDOUT.")
    parser.add_argument('-d', '--feature_delimiters', nargs='+', type=str, default=[' '],
                        help="Delimiters between the processed feature strings, to split on. Default space.")
    parser.add_argument('-s', '--skip_header', nargs='?', type=bool, default=False,
                        help="Skip the first line of the input (usually a header). Default false.")
    args = parser.parse_args()
    create_cooccurrence_matrix(args)


if __name__ == "__main__":
    main()
