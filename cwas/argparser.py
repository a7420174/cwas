import argparse
from pathlib import Path

import dotenv


def start() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments for Initializing a CWAS workspace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_workspace = Path.home() / ".cwas"
    result.add_argument(
        "-w",
        "--workspace",
        dest="workspace",
        required=False,
        type=Path,
        default=default_workspace,
        help="Path to your CWAS workspace directory",
    )
    return result

def configuration() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments for CWAS Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument(
        "-d",
        "--annotation_data_dir",
        dest="data_dir",
        required=False,
        type=Path,
        help="Path to your annotation data directory",
    )
    result.add_argument(
        "-m",
        "--gene_matrix",
        dest="gene_matrix",
        required=False,
        type=Path,
        help="Path to your gene matrix",
    )
    result.add_argument(
        "-a",
        "--annotation_key_config",
        dest="annot_key_conf",
        required=False,
        type=Path,
        help="Path to a configuration file (.yaml) that "
        "specifies the annotation key of each "
        "annotation data file",
    )
    result.add_argument(
        "-v",
        "--vep",
        dest="vep",
        required=False,
        type=Path,
        help="Path to Variant Effect Predictor (VEP)",
    )
    result.add_argument(
        "-vrd",
        "--vep_resource_dir",
        dest="vep_resource_dir",
        required=False,
        type=Path,
        help="Path to your VEP resource directory",
    )
    return result

def preparation() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments for Annotation Data Preparation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument(
        "-p",
        "--num_proc",
        dest="num_proc",
        required=False,
        type=int,
        default=1,
        help="Max No. processes for this step",
    )
    result.add_argument(
        "-f",
        "--force_overwrite",
        dest="force_overwrite",
        action="store_const",
        const=1,
        default=0,
        help="Force to overwrite the result",
    )
    return result


def annotation() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments of CWAS annotation step",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_workspace = dotenv.dotenv_values(dotenv_path=Path.home() / ".cwas_env").get("CWAS_WORKSPACE")
    result.add_argument(
        "-v",
        "--vcf_file",
        dest="vcf_path",
        required=True,
        type=Path,
        help="Target VCF file",
    )
    result.add_argument(
        "-o_dir",
        "--output_directory",
        dest="output_dir_path",
        required=False,
        default=default_workspace,
        type=Path,
        help="Directory where output file will be saved",
    )
    result.add_argument(
        "-p",
        "--num_proc",
        dest="num_proc",
        required=False,
        type=int,
        help="Number of worker processes used to fork when running VEP",
        default=1,
    )
    return result


def categorization() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments of CWAS categorization step",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_workspace = dotenv.dotenv_values(dotenv_path=Path.home() / ".cwas_env").get("CWAS_WORKSPACE")
    result.add_argument(
        "-i",
        "--input_file",
        dest="input_path",
        required=True,
        type=Path,
        help="Annotated VCF file",
    )
    result.add_argument(
        "-o_dir",
        "--output_directory",
        dest="output_dir_path",
        required=False,
        default=default_workspace,
        type=Path,
        help="Directory where output file will be saved",
    )
    result.add_argument(
        "-p",
        "--num_proc",
        dest="num_proc",
        required=False,
        type=int,
        help="Number of worker processes for the categorization",
        default=1,
    )
    return result


def binomial_test() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments of Burden Tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_workspace = dotenv.dotenv_values(dotenv_path=Path.home() / ".cwas_env").get("CWAS_WORKSPACE")
    result.add_argument(
        "-i",
        "--input_file",
        dest="cat_path",
        required=True,
        type=Path,
        help="Categorized file",
    )
    result.add_argument(
        "-o_dir",
        "--output_directory",
        dest="output_dir_path",
        required=False,
        default=default_workspace,
        type=Path,
        help="Directory where output file will be saved",
    )
    result.add_argument(
        "-s",
        "--sample_info",
        dest="sample_info_path",
        required=True,
        type=Path,
        help="File listing information of your samples",
    )
    result.add_argument(
        "-a",
        "--adjustment_factor",
        dest="adj_factor_path",
        required=False,
        default=None,
        type=Path,
        help="File listing adjustment factors of each sample",
    )
    result.add_argument(
        "-u",
        "--use_n_carrier",
        dest="use_n_carrier",
        required=False,
        action="store_true",
        help="Use the number of samples with variants in each category for burden test instead of the number of variants",
    )
    return result


def permutation_test() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments of Burden Tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_workspace = dotenv.dotenv_values(dotenv_path=Path.home() / ".cwas_env").get("CWAS_WORKSPACE")
    result.add_argument(
        "-i",
        "--input_file",
        dest="cat_path",
        required=True,
        type=Path,
        help="Categorized file (gzipped)",
    )
    result.add_argument(
        "-o_dir",
        "--output_directory",
        dest="output_dir_path",
        required=False,
        default=default_workspace,
        type=Path,
        help="Directory where output file will be saved",
    )
    result.add_argument(
        "-s",
        "--sample_info",
        dest="sample_info_path",
        required=True,
        type=Path,
        help="File listing information of your samples",
    )
    result.add_argument(
        "-a",
        "--adjustment_factor",
        dest="adj_factor_path",
        required=False,
        default=None,
        type=Path,
        help="File listing adjustment factors of each sample",
    )
    result.add_argument(
        "-n",
        "--num_perm",
        dest="num_perm",
        default=10000,
        type=int,
        help="The number of label-swapping permutations",
    )
    result.add_argument(
        "-p",
        "--num_proc",
        dest="num_proc",
        required=False,
        type=int,
        help="Number of worker processes for the categorization",
        default=1,
    )
    result.add_argument(
        "-b",
        "--burden_shift",
        dest="burden_shift",
        required=False,
        action="store_true",
        help="Generate a file of binomial p-values for each burden-shifted data",
        )
    result.add_argument(
        "-rr",
        "--perm_rr",
        dest="save_perm_rr",
        required=False,
        action="store_true",
        help="Generate a file of relative risks (RRs) for each burden-shifted data",
    )
    result.add_argument(
        "-u",
        "--use_n_carrier",
        dest="use_n_carrier",
        required=False,
        action="store_true",
        help="Use the number of samples with variants in each category for burden test instead of the number of variants",
    )
    return result

def extract_variant() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments of Burden Tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_workspace = dotenv.dotenv_values(dotenv_path=Path.home() / ".cwas_env").get("CWAS_WORKSPACE")
    result.add_argument(
        "-i",
        "--input_file",
        dest="input_path",
        required=True,
        type=Path,
        help="Annotated VCF file",
    )
    result.add_argument(
        "-o_dir",
        "--output_directory",
        dest="output_dir_path",
        required=False,
        default=default_workspace,
        type=Path,
        help="Directory where output file will be saved",
    )
    result.add_argument(
        "-t",
        "--tag",
        dest="tag",
        required=False,
        default=None,
        type=str,
        help="Tag used for the name of the output file (i.e., output.<tag>.extracted_variants.txt.gz)",
    )
    result.add_argument(
        "-c",
        "--category_set_path",
        dest="category_set_path",
        required=False,
        default=None,
        type=Path,
        help="Path to a text file containing categories for extracting variants",
    )
    result.add_argument(
        "-ai",
        "--annotation_info",
        dest="annotation_info",
        required=False,
        default=False,
        action="store_true",
        help="Save with annotation information attached (such as gene list, functional annotations, etc)",
    )    
    return result

def cross_category_burden() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Arguments of cross-category burden test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_workspace = dotenv.dotenv_values(dotenv_path=Path.home() / ".cwas_env").get("CWAS_WORKSPACE")
    result.add_argument(
        "-r",
        "--test_result",
        dest="test_result_path",
        required=True,
        type=Path,
        help="The path of the burden test result file used for getting the number of significant categories",
    )
    result.add_argument(
        "-o_dir",
        "--output_directory",
        dest="output_dir_path",
        required=False,
        default=default_workspace,
        type=Path,
        help="Directory where output file will be saved",
    )
    result.add_argument(
        "-b",
        "--burden_shift",
        dest="burden_shift_path",
        required=True,
        type=Path,
        help=("The path of a file of binomial p values for each burden-shifted data\n"
                "The file is generated by the 'cwas permutation_test' command."),
    )
    result.add_argument(
        "-t",
        "--target",
        dest="target_category_path",
        required=False,
        type=Path,
        help=("The path of an annotation term list file\n"
                "You can combine category sets for two or more terms using ','"
                "If the file is not provided, all annotation terms from $CATEGORY_DOMAIN are used"),
    )
    result.add_argument(
        "-c",
        "--cutoff",
        dest="cutoff",
        type=int,
        default=0,
        help="The number of mutations in a category to be included in the test"
    )
    result.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        help="Plot the results",
    )
    result.add_argument(
        "-n",
        "--noncoding",
        dest="noncoding",
        action="store_true",
        help="Use noncoding categories",
    )
    return result