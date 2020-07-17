#!/usr/bin/env python
"""
Script for data preparation for category-wide association study (CWAS)
#TODO: Multiprocessing
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pysam
import yaml

from utils import get_curr_time, execute_cmd, bgzip_tabix


def main():
    print(__doc__)

    # Paths for this script
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(curr_dir)
    ori_file_conf_path = os.path.join(project_dir, 'conf', 'download_filepaths.yaml')  # Files downloaded already
    target_file_conf_path = os.path.join(project_dir, 'conf', 'prepare_filepaths.yaml')  # Files that will be made

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    # Parse the configuration files
    ori_filepath_dict = {}
    target_filepath_dict = {}

    with open(ori_file_conf_path) as ori_file_conf_file, open(target_file_conf_path) as target_file_conf_file:
        ori_filepath_conf = yaml.safe_load(ori_file_conf_file)
        target_filepath_conf = yaml.safe_load(target_file_conf_file)

        for file_key in ori_filepath_conf[args.step]:
            ori_filepath_dict[file_key] = \
                os.path.join(project_dir, ori_filepath_conf[args.step][file_key])

        for file_key in target_filepath_conf[args.step]:
            target_filepath_dict[file_key] = \
                os.path.join(project_dir, target_filepath_conf[args.step][file_key])

    if args.step == 'simulation':
        print(f'[{get_curr_time()}, Progress] Prepare data for simulating random mutation')
        sort_gap_region(ori_filepath_dict['gap'], target_filepath_dict['gap'])
        make_mask_region(ori_filepath_dict['lcr'], target_filepath_dict['gap'], target_filepath_dict['mask_region'])

        chroms = [f'chr{n}' for n in range(1, 23)]
        for chrom in chroms:
            mask_fasta(ori_filepath_dict[chrom], target_filepath_dict[chrom], target_filepath_dict['mask_region'])

        chr_fa_paths = [target_filepath_dict[chrom] for chrom in chroms]
        create_chrom_size_list(chr_fa_paths, target_filepath_dict['chrom_size'])

    elif args.step == 'annotation':
        print(f'[{get_curr_time()}, Progress] Prepare data for variant annotation')

        # Filter entries of Yale PsychENCODE BED files
        file_keys = ['Yale_H3K27ac_CBC', 'Yale_H3K27ac_DFC']
        for file_key in file_keys:
            filt_yale_bed(ori_filepath_dict[file_key], target_filepath_dict[file_key])
            bgzip_tabix(target_filepath_dict[file_key])

        # Path settings for this step
        annot_dir = os.path.join(project_dir, 'data', 'annotate')
        annot_conf_path = os.path.join(project_dir, 'conf', 'annotation.yaml')
        annot_bed_path_dict = {}  # Dictionary of custom BED file paths

        # Parse the configuration file for this step
        with open(annot_conf_path) as annot_conf_file:
            annot_filename_dict = yaml.safe_load(annot_conf_file)

            for annot_key in annot_filename_dict:
                annot_bed_path_dict[annot_key] = os.path.join(annot_dir, annot_filename_dict[annot_key])

        # Merge annotation information of the custom BED files
        merge_bed_path = os.path.join(annot_dir, 'merged_annotation.bed')
        merge_annot(merge_bed_path, annot_bed_path_dict)

    print(f'[{get_curr_time()}, Progress] Done')


def create_arg_parser() -> argparse.ArgumentParser:
    """ Create an argument parser for this script and return it """
    # Create a top-level argument parser
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(description='Step for which data is prepared', dest='step')
    subparsers.add_parser('simulation', description='Prepare data to simulate random mutations',
                          help='Prepare data to simulate random mutations (arg "simulate -h" for usage)')
    subparsers.add_parser('annotation', description='Prepare data to annotate variants',
                          help='Prepare data to annotate variants (arg "annotate -h" for usage)')
    return parser


def sort_gap_region(in_gap_path: str, sort_gap_path: str):
    """ Sort a list of gap regions using the UNIX command and create a BED file listing sorted gap regions """
    if os.path.isfile(sort_gap_path):
        print(f'[{get_curr_time()}, Progress] There is already a list of sorted gap regions so skip this step')
    else:
        cmd = f'gunzip -c {in_gap_path} | cut -f2,3,4 - | sort -k1,1 -k2,2n | gzip > {sort_gap_path};'
        print(f'[{get_curr_time()}, Progress] Sort the gap regions')
        execute_cmd(cmd)


def make_mask_region(lcr_path: str, sort_gap_path: str, mask_region_path: str):
    """ Create a BED file listing masked regions by merging gap and LCR regions """
    if os.path.isfile(mask_region_path):
        print(f'[{get_curr_time()}, Progress] Masked regions have already made so skip this step')
    else:
        cmd = f'zcat {sort_gap_path} {lcr_path} | sortBed -i stdin | gzip > {mask_region_path}'
        print(f'[{get_curr_time()}, Progress] Make masked regions by merging the gap and LCR regions')
        execute_cmd(cmd)


def mask_fasta(in_fa_path: str, out_fa_path: str, mask_region_path: str):
    """ Mask regions of the input FASTA file and save the result as a new FASTA file """
    if os.path.isfile(out_fa_path):
        print(f'[{get_curr_time()}, Progress] Masked fasta file "{out_fa_path}" already exists so skip this step')
    else:
        print(f'[{get_curr_time()}, Progress] Mask the fasta file "{in_fa_path}" and index the output')
        if in_fa_path.endswith('.gz'):  # Gzipped FASTA file that must be gunzipped
            cmd = f'gunzip {in_fa_path};'
            in_fa_path = in_fa_path.replace('.gz', '')
        else:
            cmd = ''

        cmd += f'maskFastaFromBed -fi {in_fa_path} -fo {out_fa_path} -bed {mask_region_path};'
        cmd += f'samtools faidx {out_fa_path};'
        execute_cmd(cmd)


def create_chrom_size_list(chr_fa_paths: list, out_txt_path: str):
    """ Create a txt file listing the sizes total, mapped, AT/GC, and effective sizes of each chromosome """
    if os.path.isfile(out_txt_path):
        print(f'[{get_curr_time()}, Progress] A file listing total, mapped, AT/GC, and effective sizes already exists '
              f'so skip this step')
    else:
        print(f'[{get_curr_time()}, Progress] Create a file listing total, mapped, AT/GC, and effective sizes '
              f'of each chromosome')
        with open(out_txt_path, 'w') as outfile:
            print('Chrom', 'Size', 'Mapped', 'AT', 'GC', 'Effective', sep='\t', file=outfile)

            for chr_fa_path in chr_fa_paths:
                print(f'[{get_curr_time()}, Progress] Write sizes of "{chr_fa_path}"')
                fa_idx_path = chr_fa_path + '.fai'

                with open(fa_idx_path) as fa_idx_file:
                    line = fa_idx_file.read()
                    fields = line.split('\t')
                    chrom = fields[0]
                    chrom_size = int(fields[1])
                    base_start_idx = int(fields[2])

                with open(chr_fa_path, 'r') as mask_fa_file:
                    base_cnt_dict = defaultdict(int)
                    mask_fa_file.seek(base_start_idx)
                    seq = mask_fa_file.read()

                    for base in seq:
                        base_cnt_dict[base.upper()] += 1

                map_size = chrom_size - base_cnt_dict['N']
                at_size = base_cnt_dict['A'] + base_cnt_dict['T']
                gc_size = base_cnt_dict['G'] + base_cnt_dict['C']
                effect_size = (at_size / 1.75) + gc_size
                print(chrom, chrom_size, map_size, at_size, gc_size, effect_size, sep='\t', file=outfile)


def filt_yale_bed(in_bed_path: str, out_bed_path: str):
    """ Filter entries of a BED file from Yale (PsychENCODE Consortium) and
    write a BED file listing the filtered entries.
    """
    if os.path.isfile(out_bed_path) or os.path.isfile(out_bed_path + '.gz'):
        print(f'[{get_curr_time()}, Progress] Filtered bed file "{out_bed_path}" already exists '
              f'so skip this filtering step')
    else:
        print(f'[{get_curr_time()}, Progress] Filter the bed file "{in_bed_path}"')
        with pysam.TabixFile(in_bed_path) as in_bed_file, open(out_bed_path, 'w') as out_bed_file:
            for bed_fields in in_bed_file.fetch(parser=pysam.asTuple()):
                bed_val = int(bed_fields[3].split('_')[0])

                if bed_val > 1:
                    print(*bed_fields, sep='\t', file=out_bed_file)


def merge_annot(out_bed_path: str, bed_path_dict: dict):
    """ Merge all annotation information of coordinates from all the annotation BED files into one file.

    For example, assume that following three coordinates exist in bed file A, B and C, respectively.
    - chr1  100 300 1
    - chr1  200 400 1
    - chr1  250 350 1

    If these coordinates are merged, than following coordinates will be newly produced.
    - chr1  100 200 annotation A
    - chr1  200 250 annotation A, B
    - chr1  250 300 annotation A, B, C
    - chr1  300 350 annotation B, C
    - chr1  350 400 annotation B

    The information of annotation for each coordinate will be represented as an annotation integer, which is a one-hot
    encoding of annotation information. In this case, A and C are represented as a least significant bit and a most
    significant bit, respectively, For example, if all of the annotation files are A, B, and C, then an annotation
    integer of the second coordinate above is represented as 6(b'011).

    So here is a final result.
    - chr1  100 200 1
    - chr1  200 250 3
    - chr1  250 300 7
    - chr1  300 350 6
    - chr1  350 400 2

    """
    if os.path.isfile(out_bed_path):
        print(f'[{get_curr_time()}, Progress] A annotation-merged bed file already exists so skip the merge step.')
    else:
        print(f'[{get_curr_time()}, Progress] Merge all annotation information of all the input annotation BED files')
        chroms = [f'chr{n}' for n in range(1, 23)]
        chr_merge_bed_paths = [out_bed_path.replace('.bed', f'.{chrom}.bed') for chrom in chroms]

        # Make a merged BED file for each chromosome
        for i in range(len(chroms)):
            print(f'[{get_curr_time()}, Progress] Merge for {chroms[i]}')
            merge_annot_by_chrom(chroms[i], chr_merge_bed_paths[i], bed_path_dict)

        # Write headers of the merged bed file
        print(f'[{get_curr_time()}, Progress] Create a BED file with merged annotation information')
        with open(out_bed_path, 'w') as outfile:
            annot_key_str = '|'.join(bed_path_dict.keys())
            print(f'#ANNOT={annot_key_str}', file=outfile)
            print('#chrom', 'start', 'end', 'annot_int', sep='\t', file=outfile)

        # Append the merged BED file of each chromosome
        for i in range(len(chroms)):
            cmd = f'cat {chr_merge_bed_paths[i]} >> {out_bed_path};'
            cmd += f'rm {chr_merge_bed_paths[i]};'
            execute_cmd(cmd)

        bgzip_tabix(out_bed_path)


def merge_annot_by_chrom(chrom: str, out_bed_path: str, bed_path_dict: dict):
    """ Merge annotation information of all BED coordinates of one chromosome from all the BED files """
    start_to_key_idx = {}
    end_to_key_idx = {}
    pos_set = set()

    # Read coordinates from each annotation bed file
    for i, annot_bed_key in enumerate(bed_path_dict.keys()):
        annot_bed_path = bed_path_dict[annot_bed_key]
        with pysam.TabixFile(annot_bed_path) as bed_file:
            for fields in bed_file.fetch(chrom, parser=pysam.asTuple()):
                start = int(fields[1])
                end = int(fields[2])

                # Init
                if start_to_key_idx.get(start) is None:
                    start_to_key_idx[start] = []

                if end_to_key_idx.get(end) is None:
                    end_to_key_idx[end] = []

                start_to_key_idx[start].append(i)
                end_to_key_idx[end].append(i)
                pos_set.add(start)
                pos_set.add(end)

    # Create a BED file listing new coordinates with merged annotation information
    pos_list = sorted(pos_set)
    annot_cnt = np.zeros(len(bed_path_dict.keys()))
    prev_pos = -1
    n_bed = 0

    with open(out_bed_path, 'w') as outfile:
        for pos in pos_list:
            if n_bed > 0:  # Make a new coordinate
                one_hot = np.vectorize(lambda x: 1 if x else 0)(annot_cnt)
                annot_int = one_hot_to_int(one_hot)
                bed_entry = (chrom, prev_pos, pos, annot_int)
                print(*bed_entry, sep='\t', file=outfile)

            end_key_ind = end_to_key_idx.get(pos)
            start_key_ind = start_to_key_idx.get(pos)

            if end_key_ind is not None:  # This position is an end of at least one bed coordinate.
                for end_key_idx in end_key_ind:
                    annot_cnt[end_key_idx] -= 1
                n_bed -= len(end_key_ind)

            if start_key_ind is not None:  # This position is a start of at least one bed coordinate.
                for start_key_idx in start_key_ind:
                    annot_cnt[start_key_idx] += 1
                n_bed += len(start_key_ind)

            prev_pos = pos


def one_hot_to_int(one_hot: np.ndarray) -> int:
    n = 0

    for i in range(len(one_hot)):
        if one_hot[i]:
            n += 2 ** i

    return n


if __name__ == '__main__':
    main()
