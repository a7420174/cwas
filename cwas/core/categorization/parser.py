"""
This module includes functions for parsing files used in the CWAS
categorization step. By parsing those files, these functions make the
pandas.DataFrame objects that can be directly used in the categorization
algorithm.
"""
from io import TextIOWrapper
import pathlib
import re, gzip, os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext
from multiprocessing import cpu_count
from cwas.core.common import int_to_bit_arr
from cwas.utils.log import print_err, print_log

import pyspark.sql as ps
from pyspark.sql.functions import udf

# TODO: Make the code much clearer
def parse_annotated_vcf(vcf_path: pathlib.Path) -> ps.DataFrame:
    """ Parse a Variant Calling File (VCF) that includes Variant Effect
    Predictor (VEP) and CWAS annotation information and make a
    pandas.DataFrame object listing annotated variants.
    """
    conf = SparkConf()
    conf.set('spark.driver.memory', '200g')\
        .set("spark.driver.maxResultSize", '100g')
    # Pandas API on Spark automatically uses this Spark context with the configurations set.
    SparkContext(conf=conf)
    
    spark = ps.SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.files.minPartitionNum", cpu_count()*4)

    variant_col_names = []
    csq_field_names = []  # CSQ is information from VEP
    annot_field_names = []  # Custom annotation field names

    # Read and parse the input VCF
    if vcf_path.suffix == ".gz":
        vep_vcf_file = gzip.open(vcf_path, "rt")
    else:
        vep_vcf_file = vcf_path.open("r")
    for line in vep_vcf_file:
        if line.startswith("#"):  # Comments
            if line.startswith("##INFO=<ID=CSQ"):
                csq_field_names = _parse_vcf_info_field(line)
            elif line.startswith("##INFO=<ID=ANNOT"):
                annot_field_names = _parse_annot_field(line)
            elif line.startswith("#CHROM"):
                variant_col_names = _parse_vcf_header_line(line)
        else:  # Rows of variant information follow the comments.
            assert variant_col_names, "The VCF does not have column names."
            assert csq_field_names, "The VCF does not have CSQ information."
            assert annot_field_names, (
                "The VCF does not have annotation " "information."
            )
            break
    vep_vcf_file.close()
    
    schema = ', '.join([f'{col} INT' if col == 'POS' else f'{col} STRING' for col in variant_col_names])
    result = spark.read.csv(str(vcf_path), sep='\t', header=False, comment='#', schema=schema)

    print_log("INFO", "The number of partitions: {}".format(result.rdd.getNumPartitions()))
    try:
        result = _parse_info_column(
            result, csq_field_names, annot_field_names
        )
    except KeyError:
        print_err(
            "The VCF does not have INFO column or "
            "the INFO values do not have expected field keys."
        )
        raise

    pdf_result = result.toPandas()
    spark.stop()
    
    return pdf_result


def _parse_vcf_info_field(line):
    csq_line = line.rstrip('">\n')
    info_format_start_idx = re.search(r"Format: ", csq_line).span()[1]
    csq_field_names = csq_line[info_format_start_idx:].split("|")

    return csq_field_names


def _parse_annot_field(line):
    annot_line = line.rstrip('">\n')
    annot_field_str_idx = re.search(r"Key=", annot_line).span()[1]
    annot_field_names = annot_line[annot_field_str_idx:].split("|")

    return annot_field_names


def _parse_vcf_header_line(line):
    variant_col_names = line[1:].rstrip("\n").split("\t")
    return variant_col_names


def _parse_info_column(
    df: ps.DataFrame, csq_field_names: list, annot_field_names: list
) -> pd.DataFrame:
    """ Parse the INFO column and make a pd.DataFrame object """
    _parse_csq_column_with_list = udf(lambda csq: _parse_csq_column(csq, csq_field_names),
                                      ps.types.MapType(ps.types.StringType(), ps.types.StringType()))
    _parse_annot_column_with_list = udf(lambda annot: _parse_annot_column(annot, annot_field_names), 
                                        ps.types.MapType(ps.types.StringType(), ps.types.IntegerType()))
    df = df.withColumn('INFO', _parse_info_str(df['INFO']))
    info_keys = df.select(ps.functions.map_keys("INFO").alias("keys")).take(1)[0]['keys']
    df = df.withColumns({key: df['INFO'][key] for key in info_keys})
    df = df.withColumns({'CSQ': _parse_csq_column_with_list(df['CSQ']),
                         'ANNOT': _parse_annot_column_with_list(df['ANNOT'].cast(ps.types.IntegerType()))})
    df = df.withColumns({key: df['CSQ'][key] for key in csq_field_names})
    df = df.withColumns({key: df['ANNOT'][key] for key in annot_field_names})
    df = df.drop("INFO", "CSQ", "ANNOT")

    return df

@udf(returnType=ps.types.MapType(ps.types.StringType(), ps.types.StringType()))
def _parse_info_str(info_str: str) -> dict:
    """ Parse the string of the INFO field to make a dictionary """
    info_dict = {}
    key_value_pairs = info_str.split(";")

    for key_value_pair in key_value_pairs:
        key, value = key_value_pair.split("=", 1)
        info_dict[key] = value

    return info_dict

def _parse_csq_column(
    csq_field: str, csq_field_names: list
) -> dict:
    """ Parse the string of the INFO field to make a dictionary """
    csq_dict = {}
    csq_record = csq_field.split("|")

    for key, value in zip(csq_field_names, csq_record):
        csq_dict[key] = value

    return csq_dict

def _parse_annot_column(
    annot_int: int, annot_field_names: list
) -> dict:
    """ Parse the string of the INFO field to make a dictionary """
    annot_dict = {}
    annot_field_cnt = len(annot_field_names)
    annot_record = int_to_bit_arr(annot_int, annot_field_cnt)
    for key, value in zip(annot_field_names, annot_record):
        annot_dict[key] = int(value)
        
    return annot_dict


def parse_gene_matrix(gene_matrix_path: pathlib.Path) -> dict:
    """ Parse the gene matrix file and make a dictionary.
    The keys and values of the dictionary are gene symbols
    and a set of type names where the gene is associated,
    respectively.
    """
    with gene_matrix_path.open("r") as gene_matrix_file:
        return _parse_gene_matrix(gene_matrix_file)


def _parse_gene_matrix(gene_matrix_file: TextIOWrapper) -> dict:
    result = dict()
    header = gene_matrix_file.readline()
    all_gene_types = np.array(header.rstrip("\n").split("\t")[2:])

    for line in gene_matrix_file:
        _, gene_symbol, *gene_matrix_values = line.rstrip("\n").split("\t")
        gene_types = all_gene_types[np.array(gene_matrix_values) == "1"]
        result[gene_symbol] = set(gene_types)

    return result
