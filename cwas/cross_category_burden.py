import argparse, yaml, re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from cwas.runnable import Runnable
from cwas.utils.check import check_is_file
import cwas.utils.log as log


class CrossCategoryBurden(Runnable):
    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__(args)
        self._test_result = None
        self._burden_shift_pvals = None
        self._target_categories = None
        self._result_path = None
        self._plot_dir = None

    @staticmethod
    def _print_args(args: argparse.Namespace):
        log.print_arg(
            "Burden test result file", 
            args.test_result_path
        )
        log.print_arg(
            "File of binomial p values for each burden-shifted data", 
            args.burden_shift_path
        )
        log.print_arg(
            "Target category list", 
            args.target_category_path
        )
        log.print_arg(
            "Use noncoding categories",
            args.noncoding
        )

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        check_is_file(args.test_result_path)
        check_is_file(args.burden_shift_path)
        if args.target_category_path is not None:
            check_is_file(args.target_category_path)

    @property
    def test_result_path(self) -> Path:
        return self.args.test_result_path.resolve()

    @property
    def burden_shift_path(self) -> Path:
        return self.args.burden_shift_path.resolve()

    @property
    def target_category_path(self) -> Optional[Path]:
        if self.args.target_category_path is None:
            return None
        return self.args.target_category_path.resolve()

    @property
    def output_dir_path(self) -> Path:
        return self.args.output_dir_path.resolve()

    @property
    def test_result(self) -> pd.DataFrame:
        if self._test_result is None:
            log.print_progress("Load the burden test result")
            self._test_result = pd.read_table(
                self.test_result_path, index_col="Category"
            )
            try:
                self._test_result["Count"] = self._test_result["Case_Variant_Count"] + self._test_result["Ctrl_Variant_Count"]
            except KeyError:
                self._test_result["Count"] = self._test_result["Case_Carrier_Count"] + self._test_result["Ctrl_Carrier_Count"]
        return self._test_result

    @property
    def burden_shift_pvals(self) -> pd.DataFrame:
        if self._burden_shift_pvals is None:
            log.print_progress("Load the burden-shifted p values")
            self._burden_shift_pvals = pd.read_table(
                self.burden_shift_path, index_col="Trial"
            )
        return self._burden_shift_pvals

    @property
    def target_categories(self) -> Optional[dict]:
        if self._target_categories is None:
            noncoding_terms = 'UTRsRegion IntergenicRegion IntronRegion lincRnaRegion NoncodingRegion OtherTranscriptRegion PromoterRegion SpliceSiteNoncanonRegion'.split()
            coding_terms = 'CodingRegion FrameshiftRegion InFrameRegion LoFRegion MissenseRegion DamagingMissenseRegion SilentRegion'.split()
            if self.args.noncoding:
                noncoding_idx = (self.burden_shift_pvals.columns.to_series()
                                                            .str.split("_", expand=True)[3]
                                                            .isin(noncoding_terms))
                all_cats = self.burden_shift_pvals.columns[noncoding_idx]
            else:
                all_cats = self.burden_shift_pvals.columns
                
            if self.target_category_path is None:
                category_domain_path = Path(self.get_env("CATEGORY_DOMAIN"))
                yaml_data = self._load_yaml_file(category_domain_path)
                target_terms = [term for domain in yaml_data.values() for term in domain if term not in ['All', 'Any', 'SNV', 'Indel']]
            else:
                with self.target_category_path.open() as f:
                    lines = f.readlines()
                target_terms = [line.strip() for line in lines]
            ## Subset categories per target terms
            self._target_categories = {
                term.replace(',', '-'): all_cats[
                    all_cats.str
                            .split("_")
                            .map(lambda l: np.any([(x in term.split(",")) for x in l]))
                ] for term in target_terms if term not in ['all', 'coding', 'noncoding']
            }
            if 'all' in target_terms:
                self._target_categories['all'] = all_cats
            if 'coding' in target_terms:
                coding_idx = (all_cats.to_series()
                                      .str.split("_", expand=True)[3]
                                      .isin(coding_terms))
                self._target_categories['coding'] = all_cats[coding_idx]
            if 'noncoding' in target_terms:
                noncoding_idx = (all_cats.to_series()
                                         .str.split("_", expand=True)[3]
                                         .isin(noncoding_terms))
                self._target_categories['noncoding'] = all_cats[noncoding_idx]
            self._target_categories = {
                term: self.test_result[self.test_result.index.isin(cats) & (self.test_result["Count"] >= self.args.cutoff)].index
                    for term, cats in self._target_categories.items()
            }
            N_cats = {term: len(self._target_categories[term])
                            for term in self._target_categories.keys()}
            log.print_log(
                "LOG", "The number of categories\n{}"
                .format(N_cats)
            )
            if np.any([N_cats[cat] == 0 for cat in N_cats]):
                log.print_warn(
                    "The number of categories is zero for some terms."
                )
                for cat in N_cats:
                    if N_cats[cat] == 0:
                        del self._target_categories[cat]
        return self._target_categories

    @staticmethod
    def _load_yaml_file(yaml_file_path: Path):
        try:
            with yaml_file_path.open("r") as in_yaml_f:
                yaml_data = yaml.safe_load(in_yaml_f)
        except yaml.YAMLError:
            log.print_err(f'"{yaml_file_path}" is not in a proper YAML format.')
            raise
        return yaml_data
    
    @property
    def result_path(self) -> Path:
        if self._result_path is None:
            self._result_path = Path(
                f"{self.output_dir_path}/" +
                re.sub(r'\.\w+_test\.txt', 
                       '.burdenshift_noncoding_result.txt', 
                       self.test_result_path.name)
            ) if self.args.noncoding else Path(
                f"{self.output_dir_path}/" +
                re.sub(r'\.\w+_test\.txt', 
                       '.burdenshift_result.txt', 
                       self.test_result_path.name)
            )
        return self._result_path
    
    @property
    def plot_dir(self) -> Path:
        if self._plot_dir is None:
            self._plot_dir = Path(
                f"{self.output_dir_path}/" +
                re.sub(r"_test\.txt$|_test\.txt\.gz$", 
                       "_test_plots", 
                       self.test_result_path.name)
            )
            self._plot_dir.mkdir(exist_ok=True)
        return self._plot_dir
    
    def run(self):
        self.perm_test()
        self.save_result()
        self.update_env()

    def perm_test(self):
        self._result = []
        for term, cats in self.target_categories.items():
            log.print_progress("Run permutation test for {}".format(term))
            self._result.append(
                self.perm_test_per_term(term, cats)
            )
        self._result = pd.DataFrame(
            self._result, columns=["Term", "N_cats_case", "N_cats_ctrl", "P_case", "P_ctrl"]
        ).set_index("Term")

    def perm_test_per_term(self, term: str, target_cats: pd.Index) -> list:
        ## Subset the categories for each annotation term
        test_result_trim = self.test_result.loc[target_cats]
        burden_shift_pvals_trim = self.burden_shift_pvals[target_cats]
        ## Count the number of significant categories (nominal P < 0.05) for each burden-shifted data
        nobs_case = ((test_result_trim["P"] <= 0.05) & (test_result_trim["Relative_Risk"] > 1)).sum()
        nobs_ctrl = ((test_result_trim["P"] <= 0.05) & (test_result_trim["Relative_Risk"] < 1)).sum()
        ## NOTICE: If P value is below 0, the relative risk is less than 1.
        ncats_case_perm = ((burden_shift_pvals_trim.abs() <= 0.05) & (burden_shift_pvals_trim > 0)).sum(axis=1)
        ncats_ctrl_perm = ((burden_shift_pvals_trim.abs() <= 0.05) & (burden_shift_pvals_trim < 0)).sum(axis=1)
        ## Calculate the p values
        pval_case = (ncats_case_perm >= nobs_case).sum() / len(ncats_case_perm)
        pval_ctrl = (ncats_ctrl_perm >= nobs_ctrl).sum() / len(ncats_ctrl_perm)
        ## Plotting
        if self.args.plot:
            import matplotlib.pyplot as plt
            import matplotlib.transforms as transforms
            import seaborn as sns
            fig, ax = plt.subplots(dpi=300)
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            sns.kdeplot(
                x = np.concatenate([ncats_case_perm, ncats_ctrl_perm]),
                fill=True, color="gray", ax=ax
            )
            plt.xlabel("Number of nominally significant categories")
            plt.axvline(nobs_case, color='#e41a1c', linestyle='--', linewidth=1)
            plt.axvline(nobs_ctrl, color='#377eb8', linestyle='--', linewidth=1)
            plt.text(nobs_case, 0.3, "{} categories in\ncases, P = {}".format(nobs_case, pval_case), color='#e41a1c', transform=trans)
            plt.text(nobs_ctrl, 0.15, "{} categories in\ncontrols, P = {}".format(nobs_ctrl, pval_ctrl), color='#377eb8', transform=trans)
            plot_path = self.plot_dir / f"{term}_perm.png"
            plt.savefig(
                plot_path
            )
            plt.close()

            # plt.subplots(dpi=300)
            # sns.kdeplot(
            #     x = ncats_case_perm,
            #     fill=True, color="gray"
            # )
            # plt.xlabel("Number of nominally significant categories")
            # plt.axvline(nobs_case, color='#e41a1c', linestyle='--', linewidth=1)
            # plot_path = self.plot_dir / f"{term}_case_perm.png"
            # plt.savefig(
            #     plot_path
            # )
            # plt.close()

            # plt.subplots(dpi=300)
            # sns.kdeplot(
            #     x = ncats_ctrl_perm,
            #     fill=True, color="gray"
            # )
            # plt.xlabel("Number of nominally significant categories")
            # plt.axvline(nobs_ctrl, color='#377eb8', linestyle='--', linewidth=1)
            # plot_path = self.plot_dir / f"{term}_ctrl_perm.png"
            # plt.savefig(
            #     plot_path
            # )
            # plt.close()
            
        ## Make a result table
        return [term, nobs_case, nobs_ctrl, pval_case, pval_ctrl]
        
    def save_result(self):
        log.print_progress(f"Save the result to the file {self.result_path}")
        self._result.to_csv(self.result_path, sep="\t")

    def update_env(self):
        self.set_env("BURDENSHIFT_RESULT", self.result_path)
        self.save_env()
