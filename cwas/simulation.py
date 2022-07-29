import multiprocessing as mp
import pandas as pd
import numpy as np
import yaml, os, gzip, sys, argparse, pickle
from functools import cached_property, partial
from pathlib import Path
import cwas.utils.log as log
from cwas.core.categorization.parser import (
    parse_annotated_vcf,
)
from cwas.core.common import cmp_two_arr
from cwas.utils.check import check_is_file, check_num_proc, check_same_n_lines
from cwas.core.simulation.fastafile import FastaFile
from cwas.core.simulation.randomize import label_variant, pick_mutation
from scipy.stats import binom_test, norm
from cwas.runnable import Runnable
from cwas.annotation import Annotation
from cwas.categorization import Categorization
from cwas.binomial_test import BinomialTest
from typing import Optional



class Simulation(Runnable):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._in_vcf = None
        self._sample_info = None
        self._fam_to_label_cnt = None
        self._fam_to_sample_set = None
        self._filepath_dict = None
        self._chrom_size_df = None
        self._fa_file_dict = None
        self._rand_mut_paths = None
        self._annot_vcf_paths = None
        self._cat_result_paths = None
        self._burden_test_paths = None
        self._zscore_df = None
        self._resume = None


    @staticmethod
    def _create_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Arguments of CWAS simulation step",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument('-i', '--in_vcf', dest='in_vcf_path', required=False, type=Path,
                            help='Input VCF file which is referred to generate random mutations '
                                 '(Default: $ANNOTATED_VCF)')
        parser.add_argument('-s', '--sample_info', dest='sample_info_path', required=True, type=Path,
                            help='File listing sample IDs with their families and sample_types (case or ctrl)')
        parser.add_argument('-o', '--out_dir', dest='out_dir', required=False, type=Path,
                            help='Directory of outputs that lists random mutations. '
                                'The number of outputs will be the same with the number of simulations. '
                                '(Default: $HOME/.cwas/random-mutation)')
        parser.add_argument('-t', '--out_tag', dest='out_tag', required=False, type=str,
                            help='Prefix of output files. Each output file name will start with this tag. '
                                '(Default: rand_mut)', default='rand_mut')
        parser.add_argument('-n', '--num_sim', dest='num_sim', required=False, type=int,
                            help='Number of simulations to generate random mutations (Default: 1)', default=1)
        parser.add_argument('-p', '--num_proc', dest='num_proc', required=False, type=int,
                            help='Number of processes for this script (only necessary for split VCF files) '
                                '(Default: 1)', default=1)
        parser.add_argument(
            "-a", "--adjustment_factor",
            dest="adj_factor_path", required=False, default=None, type=Path,
            help="File listing adjustment factors of each sample",
        )
        parser.add_argument(
            "-u", "--use_n_carrier",
            dest="use_n_carrier", required=False, default=False, action="store_true",
            help="Use the number of samples with variants in each category for burden test instead of the number of variants",
        )
        parser.add_argument(
            "-r", "--resume",
            dest="resume", required=False, default=False, action="store_true",
            help="Resume the simulation from the last step. Assume some generated output files are not truncated.",
        )
        return parser


    @staticmethod
    def _print_args(args: argparse.Namespace):
        log.print_arg("Input VCF file", args.in_vcf_path if args.in_vcf_path else "Not specified: $ANNOTATED_VCF will be used")
        log.print_arg("Sample information file", args.sample_info_path)
        log.print_arg("Output directory", args.out_dir)
        log.print_arg("Output tag (prefix of output files)", args.out_tag)
        log.print_arg("Number of simulations", args.num_sim)
        log.print_arg("File listing adjustment factors of each sample", args.adj_factor_path)
        log.print_arg("If the number of carriers is used for burden test or not", args.use_n_carrier)
        log.print_arg(
            "No. worker processes for simulations",
            f"{args.num_proc: ,d}",
        )
        

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        check_is_file(args.sample_info_path)
        check_num_proc(args.num_proc)
        if args.adj_factor_path is not None:
            check_is_file(args.adj_factor_path)

    @property
    def in_vcf_path(self) -> Path:
        return (
            self.args.in_vcf_path.resolve()
            if self.args.in_vcf_path
            else Path(self.get_env("ANNOTATED_VCF"))
        )
    
    @property
    def sample_info_path(self) -> Path:
        return self.args.sample_info_path.resolve()
    
    @property
    def out_dir(self) -> Path:
        return (
            self.args.out_dir.resolve()
            if self.args.out_dir
            else self.workspace / "random-mutations"
        )

    @property
    def out_tag(self) -> str:
        return self.args.out_tag

    @property
    def num_sim(self) -> int:
        return self.args.num_sim

    @property
    def num_proc(self) -> int:
        return self.args.num_proc

    @property
    def adj_factor_path(self) -> Optional[Path]:
        return (
            self.args.adj_factor_path.resolve()
            if self.args.adj_factor_path
            else None
        )

    @property
    def use_n_carrier(self) -> bool:
        return self.args.use_n_carrier

    @property
    def resume(self) -> bool:
        if self._resume is None:
            self._resume = self.args.resume
        return self._resume

    @property
    def in_vcf(self) -> pd.DataFrame:
        if not self.in_vcf_path:
            self.in_vcf_path = Path(self.get_env("ANNOTATED_VCF"))
        check_is_file(self.in_vcf_path)

        if self._in_vcf is None:
            self._in_vcf = parse_annotated_vcf(
                self.in_vcf_path
            )[["REF", "ALT", "SAMPLE"]]
        return self._in_vcf

    @property
    def sample_info(self) -> pd.DataFrame:
        if self._sample_info is None:
            self._sample_info = pd.read_table(
                self.sample_info_path, index_col="SAMPLE"
            )
        return self._sample_info

    @cached_property
    def fam_to_label_cnt(self) -> dict:
        sample_to_fam = self.sample_info.to_dict()['FAMILY']
        variant_labels = np.vectorize(label_variant)(self.in_vcf.REF.values, self.in_vcf.ALT.values)
        
        fam_to_label_cnt = {}
        for sample, variant_label in zip(self.in_vcf["SAMPLE"].values, variant_labels):
            family = sample_to_fam[sample]
            label_cnt_arr = fam_to_label_cnt.get(family, np.zeros(4, dtype=int))
            label_cnt_arr[variant_label] += 1
            fam_to_label_cnt[family] = label_cnt_arr
        
        if self._fam_to_label_cnt is None:
            self._fam_to_label_cnt = fam_to_label_cnt
            
        return self._fam_to_label_cnt

    @cached_property
    def fam_to_sample_set(self) -> dict:
        sample_to_fam = self.sample_info.to_dict()['FAMILY']
        
        fam_to_sample_set = {}
        for sample in self.in_vcf["SAMPLE"].values:
            family = sample_to_fam[sample]
            sample_set = fam_to_sample_set.get(family, set())
            sample_set.add(sample)
            fam_to_sample_set[family] = sample_set
        
        if self._fam_to_sample_set is None:
            self._fam_to_sample_set = fam_to_sample_set
            
        return self._fam_to_sample_set

    @property
    def filepath_dict(self) -> dict:
        try:
            self.sim_data_dir = Path(self.get_env("SIMULATION_DATA"))
            self.annot_data_dir = Path(self.get_env("SIMULATION_PATHS"))
        except TypeError:
            raise RuntimeError(
                "Failed to get one of CWAS environment variable."
                " Maybe you omitted to run Configuration step."
            )
        with open(self.annot_data_dir) as filepath_conf_file:
            filepath_conf = yaml.safe_load(filepath_conf_file)
            filepath_dict = {path_key: (self.sim_data_dir / filepath_conf[path_key]) for path_key in filepath_conf}
        for filepath in filepath_dict.values(): check_is_file(filepath)
        
        if self._filepath_dict is None:
            self._filepath_dict = filepath_dict
        return self._filepath_dict

    @cached_property
    def chrom_size_df(self) -> pd.DataFrame:
        if self._chrom_size_df is None:
            self._chrom_size_df = pd.read_table(
                self.filepath_dict["chrom_size"], index_col="Chrom"
            )
        return self._chrom_size_df

    @cached_property
    def fa_file_dict(self) -> dict:
        # Make a dictionary for FASTA files masked in the previous preparation step.
        unq_chroms = np.unique(self.chrom_size_df.index.values)
        fa_file_dict = {}

        for chrom in unq_chroms:
            fa_file_path = self.filepath_dict[f'{chrom}']
            fa_file_dict[chrom] = FastaFile(fa_file_path)
        
        if self._fa_file_dict is None:
            self._fa_file_dict = fa_file_dict
            
        return self._fa_file_dict

    @cached_property
    def rand_mut_paths(self) -> list:
        rand_mut_paths = []
        for i in range(self.num_sim):
            str_i = str(i+1).zfill(len(str(self.num_sim)))
            output_filename = f'{self.out_tag}.{str_i}.vcf.gz'
            output_path = self.out_dir / output_filename
            rand_mut_paths.append(output_path)
        if self._rand_mut_paths is None:
            self._rand_mut_paths = rand_mut_paths
        return self._rand_mut_paths
    
    @property
    def annot_vcf_paths(self) -> list:
        if self._annot_vcf_paths is None:
            self._annot_vcf_paths = sorted(self.out_dir.glob(f'{self.out_tag}.{"?"*len(str(self.num_sim))}.annotated.vcf'))
        return self._annot_vcf_paths
    

    @property
    def cat_result_paths(self) -> list:
        if self._cat_result_paths is None:
            self._cat_result_paths = sorted(self.out_dir.glob(f'{self.out_tag}.{"?"*len(str(self.num_sim))}.categorization_result.txt.gz'))
        return self._cat_result_paths
    
    @property
    def burden_test_paths(self) -> list:
        if self._burden_test_paths is None:
            self._burden_test_paths = sorted(self.out_dir.glob(f'{self.out_tag}.{"?"*len(str(self.num_sim))}.burden_test.txt.gz'))
        return self._burden_test_paths
    
    
    @property
    def zscore_df_path(self) -> Path:
        return Path(
            str(self.cat_result_path).replace('.categorization_result.txt', '.zscores.txt')
        )
        
    @property
    def zscore_df(self) -> pd.DataFrame:
        if self._zscore_df is None:
            self._zscore_df = pd.read_table(self.zscore_df_path, index_col='Simulation')
        return self._zscore_df

    @property
    def corr_mat_path(self) -> Path:
        return Path(
            str(self.zscore_df_path).replace('.zscores.txt', '.corr_mat.pickle')
        )

    @property
    def neg_lap_path(self) -> Path:
        return Path(
            str(self.zscore_df_path).replace('.zscores.txt', '.neg_lap.pickle')
        )
        
    @property
    def eig_val_path(self) -> Path:
        return Path(
            str(self.zscore_df_path).replace('.zscores.txt', '.eig_vals.pickle')
        )

    def run(self):
        self.simulate() 
        self.concat_zscores()
        self.get_n_etests()
        self.update_env()
        log.print_progress("Done")

    
    def simulate(self):
        self.prepare()
        self.make_rand_mut_files()
        self.annotate()
        self.categorize()
        self.burden_tests()


    def prepare(self):
        if not cmp_two_arr(self.sample_info.index, self.in_vcf['SAMPLE'].unique()):
            log.print_warn("The sample IDs in the sample info file and the VCF file are not the same.")

        log.print_progress("Make an output directory")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def make_rand_mut_files(self):
        """ Make VCF files listing random mutations"""
        log.print_progress(self.make_rand_mut_files.__doc__)
        pre_files = sorted(self.out_dir.glob(f'{self.out_tag}.{"?"*len(str(self.num_sim))}.vcf.gz'))
        if len(pre_files) == 0:
            target_files = self.rand_mut_paths
        elif len(pre_files) == self.num_sim:
            log.print_progress("Checking the number of lines for the last 100 VCFs...")
            check_same_n_lines(pre_files[-100:], gzip_file=True)
            log.print_log(
                "NOTICE",
                "You already have random mutation vcfs. Skip this step.",
                False,
            )
            return
        elif self.resume & (len(pre_files) < self.num_sim):
            log.print_progress("Checking the number of lines for the last 100 VCFs...")
            check_same_n_lines(pre_files[-100:], gzip_file=True)
            log.print_log(
                "NOTICE",
                f"You have some random mutation vcfs ({len(pre_files)}). Resume this step.",
                False,
            )
            target_files = sorted(list(set(self.rand_mut_paths) - set(pre_files)))
            self._resume = False
        else:
            raise RuntimeError(
                "The number of random mutation vcfs is not the same as the number of simulations."
                " Check and remove the files to rerun this step."
            )
            
        if self.num_proc == 1:
            for output_path in target_files:
                self.make_rand_mut_file(output_path)
        else:
            with mp.Pool(self.num_proc) as pool:
                pool.map(
                    self.make_rand_mut_file,
                    target_files
                )
        
        for fasta_file in self.fa_file_dict.values():
            fasta_file.close()


    def make_rand_mut_file(self, output_path):
        """ Make a VCF file listing random mutations """
            
        rand_variants = []

        for fam in self.fam_to_label_cnt:
            label_cnt_arr = self.fam_to_label_cnt[fam]
            sample_ids = list(self.fam_to_sample_set[fam])

            for label, label_cnt in enumerate(label_cnt_arr):
                for _ in range(label_cnt):
                    rand_variant = self.make_random_mutation(label, sample_ids)
                    rand_variants.append(rand_variant)

        rand_variants.sort(key=lambda x: (x.get('chrom'), x.get('pos')))
        
        with gzip.open(output_path, 'wt') as outfile:
            print('#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', sep='\t', file=outfile)

            for variant in rand_variants:
                print(variant['chrom'], variant['pos'], variant['id'], variant['ref'], variant['alt'],
                    variant['qual'], variant['filter'], variant['info'], sep='\t', file=outfile)
            
       
    def make_random_mutation(self, label: int, sample_ids: list) -> dict:
        """ Generate and return a random mutation in the VCF format """
        
        chrom_eff_sizes = self.chrom_size_df['Effective'].values
        chrom_probs = chrom_eff_sizes / np.sum(chrom_eff_sizes)  # Normalization
        chrom_sizes = self.chrom_size_df['Size'].values

        np.random.seed()
        sample_id = np.random.choice(sample_ids)
        ref = None
        alt = None

        while True:
            chrom_idx = np.random.choice(range(len(chrom_probs)), p=chrom_probs)
            chrom_size = chrom_sizes[chrom_idx]
            chrom = f'chr{chrom_idx + 1}'
            fa_file = self.fa_file_dict[chrom]

            pos = np.random.randint(chrom_size)
            base = fa_file.get_base(chrom, pos).upper()

            if base == 'N':
                continue

            ref, alt = pick_mutation()

            if base != ref:
                continue

            break

        alt += 'A' * label

        variant = {
            'chrom': chrom,
            'pos': pos + 1,  # 0-based -> 1-based
            'id': f'{chrom}:{pos + 1}:{ref}:{alt}',
            'ref': ref,
            'alt': alt,
            'qual': '.',
            'filter': '.',
            'info': f'SAMPLE={sample_id}'
        }
        return variant
    
    
    def annotate(self):
        """ Annotation for random mutations """
        log.print_progress(self.annotate.__doc__)
        pre_files = sorted(self.out_dir.glob(f'{self.out_tag}.{"?"*len(str(self.num_sim))}.annotated.vcf'))
        if len(pre_files) == 0:
            target_inputs = self.rand_mut_paths
        elif len(pre_files) == self.num_sim:
            log.print_progress("Checking the number of lines for the last 100 VCFs...")
            check_same_n_lines(pre_files[-100:])
            log.print_log(
                "NOTICE",
                "You already have annotated vcfs. Skip this step.",
                False,
            )
            return
        elif self.resume & (len(pre_files) < self.num_sim):
            log.print_progress("Checking the number of lines for the last 100 VCFs...")
            check_same_n_lines(pre_files[-100:])
            log.print_log(
                "NOTICE",
                f"You have some annotated vcfs ({len(pre_files)}). Resume this step.",
                False,
            )
            target_inputs = sorted(list(set(self.rand_mut_paths) - set([Path(str(path).replace('.annotated.vcf', '.vcf.gz')) for path in pre_files])))
            self._resume = False
        else:
            raise RuntimeError(
                "The number of annotated vcfs is not the same as the number of simulations."
                " Check and remove the files to rerun this step."
            )

        if self.num_proc == 1:
            for rand_mut_path in target_inputs:
                self._annotate_one(rand_mut_path)
        else:
            def mute():
                sys.stderr = open(os.devnull, 'w')  
            with mp.Pool(self.num_proc, initializer=mute) as pool:
                pool.map(
                    self._annotate_one,
                    target_inputs,
                )


    @staticmethod
    def _annotate_one(rand_mut_path: Path):
        annotator = Annotation.get_instance(['-v', str(rand_mut_path)])
        annotator.vep_output_vcf_path = str(rand_mut_path).replace('.vcf.gz', '.vep.vcf')
        annotator.annotate_using_bigwig()
        annotator.process_vep_vcf()
        annotator.annotate_using_bed()
        # os.remove(annotator.vep_output_vcf_gz_path)
        # os.remove(annotator.vep_output_vcf_gz_path + '.tbi')
    
    
    def categorize(self):
        """ Categorize random mutations """
        log.print_progress(self.categorize.__doc__)
        pre_files = sorted(self.out_dir.glob(f'{self.out_tag}.{"?"*len(str(self.num_sim))}.categorization_result.txt.gz'))
        if len(pre_files) == 0:
            target_inputs = self.annot_vcf_paths
        elif len(pre_files) == self.num_sim:
            log.print_progress("Checking the number of lines for the last 100 result files...")
            check_same_n_lines(pre_files[-100:], gzip_file=True)
            log.print_log(
                "NOTICE",
                "You already have categorization results. Skip this step.",
                False,
            )
            return
        elif self.resume & (len(pre_files) < self.num_sim):
            log.print_progress("Checking the number of lines for the last 100 result files...")
            check_same_n_lines(pre_files[-100:], gzip_file=True)
            log.print_log(
                "NOTICE",
                f"You have some categorization results ({len(pre_files)}). Resume this step.",
                False,
            )
            target_inputs = sorted(list(set(self.annot_vcf_paths) - set([Path(str(path).replace('.categorization_result.txt.gz', '.annotated.vcf')) for path in pre_files])))
            self._resume = False
        else:
            raise RuntimeError(
                "The number of categorization results is not the same as the number of simulations."
                " Check and remove the files to rerun this step."
            )

        if self.num_proc == 1:
            for annot_vcf_path in target_inputs:
                self._categorize_one(annot_vcf_path)
        else:
            def mute():
                sys.stderr = open(os.devnull, 'w')
            with mp.Pool(self.num_proc, initializer=mute) as pool:
                pool.map(
                    self._categorize_one,
                    target_inputs,
                )


    @staticmethod
    def _categorize_one(annot_vcf_path: Path):
        categorizer = Categorization.get_instance()
        categorizer.annotated_vcf = parse_annotated_vcf(annot_vcf_path)
        categorizer.result_path = Path(str(annot_vcf_path).replace('.annotated.vcf', '.categorization_result.txt.gz'))
        categorizer.categorize_vcf()
        categorizer.remove_redundant_category()
        categorizer.save_result()


    def burden_tests(self):
        """ Burden tests for random mutations """
        log.print_progress(self.burden_tests.__doc__)
        pre_files = sorted(self.out_dir.glob(f'{self.out_tag}.{"?"*len(str(self.num_sim))}.burden_test.txt.gz'))
        if len(pre_files) == 0:
            target_inputs = self.cat_result_paths
        elif len(pre_files) == self.num_sim:
            log.print_log(
                "NOTICE",
                "You already have burden test results. Skip this step.",
                False,
            )
            return
        elif self.resume & (len(pre_files) < self.num_sim):
            log.print_log(
                "WARNING",
                f"You have some burden test results ({len(pre_files)}). "
                f"Resume this step. Check whether the files are truncated.",
                False,
            )
            target_inputs = sorted(list(set(self.cat_result_paths) - set([Path(str(path).replace('.burden_test.txt.gz', '.categorization_result.txt.gz')) for path in pre_files])))
            self._resume = False
        else:
            raise RuntimeError(
                "The number of burden test results is not the same as the number of simulations."
                " Check and remove the files to rerun this step."
            )

        _burden_test_partial = partial(self._burden_test, 
                                       sample_info_path=self.sample_info_path,
                                       use_n_carrier=self.use_n_carrier,
                                       adj_factor_path=self.adj_factor_path)
        
        if self.num_proc == 1:
            for cat_result_path in target_inputs:
                _burden_test_partial(
                    cat_result_path,
                )
        else:
            def mute():
                sys.stderr = open(os.devnull, 'w')
            with mp.Pool(self.num_proc, initializer=mute) as pool:
                pool.map(
                    _burden_test_partial,
                    target_inputs,
                )


    @staticmethod
    def _burden_test(cat_result_path: Path, sample_info_path: Path, adj_factor_path: Optional[Path], use_n_carrier: bool):
        argv = ['-c', str(cat_result_path), '-s', str(sample_info_path)]
        if adj_factor_path is not None:
            argv.extend(['-a', str(adj_factor_path)])
        if use_n_carrier:
            argv.append('-u')
        tester = BinomialTest.get_instance(argv=argv)
        tester.result_path = Path(str(cat_result_path).replace('.categorization_result.txt.gz', '.burden_test.txt.gz'))
        
        if tester.use_n_carrier:
            tester.count_carrier_for_each_category()
            tester.calculate_relative_risk_with_n_carrier()
        else:
            tester.count_variant_for_each_category()
            tester.calculate_relative_risk()

        tester.run_burden_test()
        tester.concat_category_info()
        tester.save_result()


    def concat_zscores(self):
        """ Concatenate zscores for each result """
        log.print_progress(self.concat_zscores.__doc__)
        
        try:
            self.cat_result_path = Path(self.get_env("CATEGORIZATION_RESULT"))
        except TypeError:
            raise RuntimeError(
                "Failed to get $CATEGORIZATION_RESULT CWAS environment variable."
                " Maybe you omitted to run Categorization step."
            )
        
        if self.zscore_df_path.is_file():
            log.print_log(
                "NOTICE",
                "You already have a Z score table. Skip this step.",
                False,
            )
            return
        with gzip.open(self.cat_result_path, mode='rt') as f:
            header = f.readline()
            header_fields = header.strip().split('\t')
            combs = header_fields[1:]

        binom_p = (self.sample_info["PHENOTYPE"] == "case").sum() / np.isin(self.sample_info["PHENOTYPE"], ["case", "ctrl"]).sum()
        default_p = binom_test(x=1, n=2, p=binom_p, alternative='greater')
        default_z = norm.ppf(1 - default_p)

        default_z_dict = {comb: default_z for comb in combs}
        
        _get_zscore_dict_partial = partial(self._get_zscore_dict, z_dict=default_z_dict)
        
        if self.num_proc == 1:
            z_dicts = []
            for burden_test_path in self.burden_test_paths:
                z_dict = _get_zscore_dict_partial(burden_test_path)
                z_dicts.append(z_dict)
        else:
            with mp.Pool(self.num_proc) as pool:
                z_dicts = pool.map(
                    _get_zscore_dict_partial,
                    self.burden_test_paths,
                )
                
        zscore_df = pd.DataFrame.from_records(z_dicts)        
        zscore_df.index.name = 'Simulation'
        zscore_df.index += 1
        
        zscore_df.to_csv(self.zscore_df_path, sep='\t')
        self._zscore_df = zscore_df

    @staticmethod
    def _get_zscore_dict(burden_test_path: Path, z_dict: dict) -> dict:
        with gzip.open(burden_test_path, mode='rt') as f:
            x = f.readline()

            for line in f:
                fields = line.strip().split('\t')

                if z_dict.get(fields[0]) is not None:
                    z_dict[fields[0]] = float(fields[6])
                    
        return z_dict
      
      
    def get_n_etests(self):
        """ Get the number of effective tests """
        log.print_progress(self.get_n_etests.__doc__)
        
        if not self.corr_mat_path.is_file():
            zscore_mat = self.zscore_df.values
            corr_mat = np.corrcoef(zscore_mat.T)
            if np.isnan(corr_mat).any():
                log.print_warn("The correlation matrix contains NaN. NaN will be replaced with 0,1.")
                for i in range(corr_mat.shape[0]):
                    if np.isnan(corr_mat[i, i]):
                        corr_mat[i, i] = 1.0
                np.nan_to_num(corr_mat, copy=False)
                
            log.print_progress("Writing the correlation matrix to file")
            pickle.dump(corr_mat, open(self.corr_mat_path, 'wb'), protocol=5)
        else:
            with self.corr_mat_path.open('rb') as f:
                corr_mat = pickle.load(f)

        if not self.neg_lap_path.is_file():
            neg_lap = np.abs(corr_mat)
            degrees = np.sum(neg_lap, axis=0)
            for i in range(neg_lap.shape[0]):
                neg_lap[i, :] = neg_lap[i, :] / np.sqrt(degrees)
                neg_lap[:, i] = neg_lap[:, i] / np.sqrt(degrees)
            log.print_progress("Writing the negative laplacian matrix to file")
            pickle.dump(neg_lap, open(self.neg_lap_path, 'wb'), protocol=5)
        else:
            with self.neg_lap_path.open('rb') as f:
                neg_lap = pickle.load(f)

        if not self.eig_val_path.is_file():
            eig_vals, _ = np.linalg.eig(neg_lap)
            log.print_progress("Writing the eigenvalues to file")
            pickle.dump(eig_vals, open(self.eig_val_path, 'wb'), protocol=5)
        else:
            with self.eig_val_path.open('rb') as f:
                eig_vals = pickle.load(f)
        
        e = 1e-12
        eig_vals = sorted(eig_vals, key=np.linalg.norm)[::-1]
        num_eig_val = self.num_sim
        clean_eig_vals = np.array(eig_vals[:num_eig_val])
        clean_eig_vals = clean_eig_vals[clean_eig_vals >= e]
        clean_eig_val_total_sum = np.sum(clean_eig_vals)
        clean_eig_val_sum = 0
        eff_num_test = 0
        
        for i in range(len(clean_eig_vals)):
            clean_eig_val_sum += clean_eig_vals[i]

            if clean_eig_val_sum / clean_eig_val_total_sum >= 0.99:
                eff_num_test = i + 1
                break
        
        log.print_log("RESULT", f"The number of effective tests is {eff_num_test}.", False)
        
        self.eff_num_test = eff_num_test
            
    def update_env(self):
        self.set_env("ZSCORE_TABLE", self.zscore_df_path)
        self.set_env("N_EFFECTIVE_TEST", self.eff_num_test)
        self.save_env()
