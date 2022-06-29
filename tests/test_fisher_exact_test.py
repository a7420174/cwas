import pandas as pd
import pytest
from cwas.fisher_exact_test import FisherExactTest


class FisherExactTestMock(FisherExactTest):
    """This class do not make outputs"""

    @property
    def use_n_carrier(self):
        return True

    def save_result(self):
        pass

    def update_env(self):
        pass


@pytest.fixture
def categorization_result():
    results = [
        {"SAMPLE": "Sample1", "A_B_C_D_E": 5, "a_b_c_d_e": 4},
        {"SAMPLE": "Sample2", "A_B_C_D_E": 10, "a_b_c_d_e": 0},
        {"SAMPLE": "Sample3", "A_B_C_D_E": 12, "a_b_c_d_e": 8},
        {"SAMPLE": "Sample4", "A_B_C_D_E": 7, "a_b_c_d_e": 11},
        {"SAMPLE": "Sample5", "A_B_C_D_E": 8, "a_b_c_d_e": 6},
        {"SAMPLE": "Sample6", "A_B_C_D_E": 15, "a_b_c_d_e": 0},
    ]
    return pd.DataFrame(results).set_index("SAMPLE")


@pytest.fixture
def sample_info():
    samples = [
        {"SAMPLE": "Sample1", "FAMILY": "F1", "PHENOTYPE": "case"},
        {"SAMPLE": "Sample2", "FAMILY": "F1", "PHENOTYPE": "ctrl"},
        {"SAMPLE": "Sample3", "FAMILY": "F2", "PHENOTYPE": "case"},
        {"SAMPLE": "Sample4", "FAMILY": "F2", "PHENOTYPE": "ctrl"},
        {"SAMPLE": "Sample5", "FAMILY": "F3", "PHENOTYPE": "ctrl"},
        {"SAMPLE": "Sample6", "FAMILY": "F3", "PHENOTYPE": "ctrl"},
    ]
    return pd.DataFrame(samples).set_index("SAMPLE")


@pytest.fixture
def adjustment_factor():
    adj_factors = [
        {"SAMPLE": "Sample1", "AdjustFactor": 2.0},
        {"SAMPLE": "Sample2", "AdjustFactor": 0.5},
        {"SAMPLE": "Sample3", "AdjustFactor": 0.25},
        {"SAMPLE": "Sample4", "AdjustFactor": 1.0},
        {"SAMPLE": "Sample5", "AdjustFactor": 0.5},
        {"SAMPLE": "Sample6", "AdjustFactor": 0.2},
    ]
    return pd.DataFrame(adj_factors).set_index("SAMPLE")


@pytest.fixture
def sample_info_other_sample():
    samples = [
        {"SAMPLE": "SampleA", "FAMILY": "F1", "PHENOTYPE": "case"},
        {"SAMPLE": "SampleB", "FAMILY": "F2", "PHENOTYPE": "ctrl"},
    ]
    return pd.DataFrame(samples).set_index("SAMPLE")


@pytest.fixture
def adjustment_factor_other_sample():
    adj_factors = [
        {"SAMPLE": "SampleA", "AdjustFactor": 2.0},
        {"SAMPLE": "SampleB", "AdjustFactor": 0.5},
    ]
    return pd.DataFrame(adj_factors).set_index("SAMPLE")


@pytest.fixture
def fisher_exact_test(
    categorization_result, sample_info, adjustment_factor,
):
    # This is not an appropriate usage.
    inst = FisherExactTestMock()
    inst._categorization_result = categorization_result
    inst._sample_info = sample_info
    inst._adj_factor = adjustment_factor
    return inst


@pytest.fixture
def fisher_exact_test_with_inconsistent_sample(
    categorization_result,
    sample_info_other_sample,
    adjustment_factor_other_sample,
):
    # This is not an appropriate usage.
    inst = FisherExactTestMock()
    inst._categorization_result = categorization_result
    inst._sample_info = sample_info_other_sample
    inst._adj_factor = adjustment_factor_other_sample
    return inst


def test_adjust_categorization_result(fisher_exact_test):
    fisher_exact_test._adjust_categorization_result()
    categorization_result = fisher_exact_test.categorization_result
    assert categorization_result.loc["Sample1"].to_list() == [10, 8]
    assert categorization_result.loc["Sample2"].to_list() == [5, 0]
    assert categorization_result.loc["Sample3"].to_list() == [3, 2]
    assert categorization_result.loc["Sample4"].to_list() == [7, 11]
    assert categorization_result.loc["Sample5"].to_list() == [4, 3]
    assert categorization_result.loc["Sample6"].to_list() == [3, 0]


def test_adjust_categorization_with_inconsistent_sample(
    fisher_exact_test_with_inconsistent_sample,
):
    with pytest.raises(ValueError):
        fisher_exact_test_with_inconsistent_sample._adjust_categorization_result()


def test_run_with_inconsistent_sample(fisher_exact_test_with_inconsistent_sample):
    with pytest.raises(ValueError):
        fisher_exact_test_with_inconsistent_sample.run()


def test_run(fisher_exact_test):
    fisher_exact_test._adjust_categorization_result()
    fisher_exact_test.run()
    assert fisher_exact_test._result is not None
    assert fisher_exact_test._result.index.name == "Category"
    expected_columns = [
        "variant_type",
        "gene_list",
        "conservation",
        "gencode",
        "region",
        "Case_Carrier_Count",
        "Ctrl_Carrier_Count",
        "Relative_Risk",
        "OR",
        "P",
        "P_1side",
    ]
    expected_index = [
        "A_B_C_D_E",
        "a_b_c_d_e",
    ]
    assert list(fisher_exact_test._result.columns.values) == expected_columns
    assert list(fisher_exact_test._result.index.values) == expected_index


def test_case_cnt(fisher_exact_test):
    assert fisher_exact_test.case_cnt == 2


def test_ctrl_cnt(fisher_exact_test):
    assert fisher_exact_test.ctrl_cnt == 4


def test_case_carrier_cnt(fisher_exact_test):
    fisher_exact_test._adjust_categorization_result()
    assert list(fisher_exact_test.case_carrier_cnt) == [2, 2]


def test_ctrl_carrier_cnt(fisher_exact_test):
    fisher_exact_test._adjust_categorization_result()
    assert list(fisher_exact_test.ctrl_carrier_cnt) == [4, 2]


def test_calculate_relative_risk(fisher_exact_test):
    fisher_exact_test._adjust_categorization_result()
    fisher_exact_test.run()
    expected_relative_risk1 = (2 / 2) / (4 / 4)  # A_B_C_D_E
    expected_relative_risk2 = (2 / 2) / (2 / 4)  # a_b_c_d_e
    assert fisher_exact_test._result["Relative_Risk"].to_list() == [
        expected_relative_risk1,
        expected_relative_risk2,
    ]


