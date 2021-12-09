#!/usr/bin/env python3
import argparse
import sys

from fip.chem import smiles2rdmol, rdmol2morgan_feature_smiles, standardize_mol


def make_fragments(args):
    with args.output as out, args.input as inputfile:
        for line in inputfile:
            line = line.strip()
            if line:
                mol = smiles2rdmol(line)
                if not mol:
                    out.write(f"Can't parse {line} as RDKit Mol\n")
                    continue
                mol = standardize_mol(mol, remove_hydrogens=(not args.explicit_hydrogen),
                                      remove_stereo=(not args.stereo))
                fragment_strings = rdmol2morgan_feature_smiles(mol, radius=args.max_radius, min_radius=args.min_radius,
                                                               all_H_explicit=args.explicit_hydrogen)
                out.write(args.fragment_delimiter.join(fragment_strings))
                out.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Script to process input SMILES, one per line, into EC fragments.")
    parser.add_argument('-i', '--input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="Path to the input file. Default STDIN.")
    parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="Path to the output file. Default STDOUT.")
    parser.add_argument('-r', '--max_radius', nargs='?', type=int, default=2,
                        help="The max radius of the generated fragments. Default 2.")
    parser.add_argument('-m', '--min_radius', nargs='?', type=int, default=0,
                        help="The min radius of the generated fragments. Default 0.")
    parser.add_argument('-d', '--fragment_delimiter', nargs='?', type=str, default=' ',
                        help="Delimiter between the generated fragment strings. Default single space.")
    parser.add_argument('-e', '--explicit_hydrogen', nargs='?', type=bool, default=False,
                        help="Whether to include explicit hydrogen in the EC fragments.")
    parser.add_argument('-s', '--stereo', nargs='?', type=bool, default=False,
                        help="Whether to use stereochemistry information for the fragments.")
    args = parser.parse_args()
    make_fragments(args)


if __name__ == "__main__":
    main()
