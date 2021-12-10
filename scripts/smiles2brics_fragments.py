#!/usr/bin/env python3
import argparse
import sys
import signal

from fip.chem import smiles2rdmol, rdmol2brics_blocs_smiles, standardize_mol, rdmol2smiles


def timeout_handler(signum, frame):
    raise TimeoutError(str(signum) + str(frame))


def make_fragments(args):
    signal.signal(signal.SIGALRM, timeout_handler)
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
                signal.alarm(args.timeout)
                try:
                    fragment_strings = rdmol2brics_blocs_smiles(mol, min_fragment_size=args.min_fragment_size)
                except TimeoutError:
                    fragment_strings = [rdmol2smiles(mol)]
                out.write(args.fragment_delimiter.join(fragment_strings))
                out.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Script to process input SMILES, one per line, into BRICS fragments.")
    parser.add_argument('-i', '--input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="Path to the input file. Default STDIN.")
    parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="Path to the output file. Default STDOUT.")
    parser.add_argument('-m', '--min_fragment_size', nargs='?', type=int, default=1,
                        help="The minimal fragment size. Default 1.")
    parser.add_argument('-d', '--fragment_delimiter', nargs='?', type=str, default=' ',
                        help="Delimiter between the generated fragment strings. Default single space.")
    parser.add_argument('-e', '--explicit_hydrogen', nargs='?', type=bool, default=False,
                        help="Whether to include explicit hydrogen in the EC fragments.")
    parser.add_argument('-s', '--stereo', nargs='?', type=bool, default=False,
                        help="Whether to use stereochemistry information for the fragments.")
    parser.add_argument('-t', '--timeout', nargs='?', type=int, default=10,
                        help="How many seconds to wait on a single compound, before it is given upon and skipped. Default 10 s.")
    args = parser.parse_args()
    make_fragments(args)


if __name__ == "__main__":
    main()
