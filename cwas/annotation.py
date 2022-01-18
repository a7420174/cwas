"""
CWAS Annotation Step

This step annotate user's VCF file using annotation data specified 
in the CWAS configuration step. This step mainly uses 
Variant Effect Predictor (VEP) to annotate user's VCF file.
"""
import argparse
from pathlib import Path

import yaml

from cwas.core.annotation.vep import VepCmdGenerator
from cwas.runnable import Runnable
from cwas.utils.check import check_is_file, check_num_proc
from cwas.utils.cmd import CmdExecutor
from cwas.utils.log import print_arg, print_log, print_progress


class Annotation(Runnable):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    @staticmethod
    def _create_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Arguments of CWAS annotation step",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "-v",
            "--vcf_file",
            dest="vcf_path",
            required=True,
            type=Path,
            help="Target VCF file",
        )
        parser.add_argument(
            "-p",
            "--num_process",
            dest="num_proc",
            required=False,
            type=int,
            help="Number of worker processes to use",
            default=1,
        )
        return parser

    @staticmethod
    def _print_args(args: argparse.Namespace):
        print_arg("Target VCF file", args.vcf_path)
        print_arg("Number of worker processes", args.num_proc)

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        check_is_file(args.vcf_path)
        check_num_proc(args.num_proc)

    @property
    def vep_cmd(self):
        vep_cmd_generator = VepCmdGenerator(
            self.get_env("VEP"), str(self.vcf_path)
        )
        vep_cmd_generator.output_vcf_path = self.vep_output_vcf_path
        for bw_path, annotation_key in self.bw_custom_annotations:
            vep_cmd_generator.add_bw_custom_annotation(bw_path, annotation_key)
        return vep_cmd_generator.cmd

    @property
    def vep_output_vcf_path(self):
        return (
            f"{self.get_env('CWAS_WORKSPACE')}/"
            f"{self.vcf_path.name.replace('.vcf', '.vep.vcf')}"
        )

    @property
    def bw_custom_annotations(self):
        with open(self.get_env("ANNOTATION_BW_KEY")) as infile:
            bw_custom_path_dict = yaml.safe_load(infile)
        annotation_data_dir = self.get_env("ANNOTATION_DATA")

        for bw_filename, bw_annotation_key in bw_custom_path_dict.items():
            yield (f"{annotation_data_dir}/{bw_filename}", bw_annotation_key)

    def run(self):
        self.annotate_using_bigwig()
        print_log("Notice", "Not implemented yet.")

    def annotate_using_bigwig(self):
        print_progress("BigWig custom annotations via VEP")
        if Path(self.vep_output_vcf_path).is_file():
            print_log(
                "NOTICE",
                "You have already done the BigWig custom annotations.",
                True,
            )
            return

        vep_bin, *vep_args = self.vep_cmd
        CmdExecutor(vep_bin, vep_args).execute_raising_err()
