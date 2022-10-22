from pathlib import Path
import numpy as np
from scipy.stats import norm

from cwas.burden_test import BurdenTest
from cwas.core.burden_test.binomial import binom_one_tail, binom_two_tail
from cwas.utils.log import print_progress


class BinomialTest(BurdenTest):
    @property
    def binom_p(self) -> float:
        return self.case_cnt / (self.case_cnt + self.ctrl_cnt)

    @property
    def result_path(self) -> Path:
        if self._result_path is None:
            self._result_path = Path(
                str(self.categorization_result_path).replace(
                    '.categorization_result.txt', '.binomial_test.txt'
                )
            )
        return self._result_path

    @result_path.setter
    def result_path(self, path: Path):
        self._result_path = path

    def run_burden_test(self):
        print_progress("Run binomial test")
        if self.use_n_carrier:
            n1 = self.case_carrier_cnt
            n2 = self.ctrl_carrier_cnt
        else:
            n1 = self.case_variant_cnt.round()
            n2 = self.ctrl_variant_cnt.round()
        self._result["P"] = np.vectorize(binom_two_tail)(
            n1, n2, self.binom_p
        )

        # Add the pseudocount(1) in order to avoid p-values of one
        self._result["P_1side"] = np.vectorize(binom_one_tail)(
            n1 + 1, n2 + 1, self.binom_p
        )
        # Set a lower limit to calculate finite z scores for p-values of one
        self._result["Z_1side"] = min(norm.ppf(1 - self._result["P_1side"].values), norm.ppf(1 - 0.9999999999999999))
