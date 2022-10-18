from pathlib import Path
import numpy as np
from scipy.stats import norm

from cwas.burden_test import BurdenTest
from cwas.core.burden_test.fisher_exact import fisher_two_tail, fisher_one_tail
from cwas.utils.log import print_progress


class FisherExactTest(BurdenTest):
    @property
    def result_path(self) -> Path:
        if self._result_path is None:
            self._result_path = Path(
                str(self.categorization_result_path).replace(
                    '.categorization_result.txt', '.fisher_test.txt'
                )
            )
        return self._result_path
    
    def run_burden_test(self):
        print_progress("Run Fisher's exact test")
        
        if not self.use_n_carrier:
            raise RuntimeError(
                "This method is only for '-u' option."
            )
        
        self._result["OR"], self._result["P"] = np.vectorize(fisher_two_tail)(
            self.case_carrier_cnt,
            self.ctrl_carrier_cnt,
            self.case_cnt - self.case_carrier_cnt,
            self.ctrl_cnt - self.ctrl_carrier_cnt,
        )

        _, self._result["P_1side"] = np.vectorize(fisher_one_tail)(
            self.case_carrier_cnt,
            self.ctrl_carrier_cnt,
            self.case_cnt - self.case_carrier_cnt,
            self.ctrl_cnt - self.ctrl_carrier_cnt,
        )