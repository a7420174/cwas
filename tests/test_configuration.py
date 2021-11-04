"""
Tests of the 'Configuration' step
"""
import random
from pathlib import Path

import pytest
from cwas.configuration import Configuration
from cwas.env import Env


@pytest.fixture(scope="module", autouse=True)
def create_cwas_env_file(cwas_env_path: Path, cwas_workspace: Path):
    with cwas_env_path.open("w") as cwas_env_file:
        print(f"CWAS_WORKSPACE={str(cwas_workspace)}", file=cwas_env_file)
    yield
    cwas_env_path.unlink()


@pytest.fixture(scope="module", autouse=True)
def set_env_path(cwas_env_path: Path):
    env = Env()
    env.set_path(cwas_env_path)


@pytest.fixture
def cwas_config(
    annotation_dir: Path,
    annotation_key_conf: Path,
    bw_cutoff_conf: Path,
    gene_matrix: Path,
):
    config = {
        "ANNOTATION_DATA_DIR": annotation_dir,
        "GENE_MATRIX": gene_matrix,
        "ANNOTATION_KEY_CONFIG": annotation_key_conf,
        "BIGWIG_CUTOFF_CONFIG": bw_cutoff_conf,
        "VEP": "VEP",
    }
    return config


@pytest.fixture
def invalid_file_path(cwas_workspace: Path):
    filename = f"invalid-{random.randint(1, 1000000)}"
    return cwas_workspace / filename


@pytest.fixture
def create_cwas_config_file(cwas_workspace, cwas_config):
    _create_cwas_config_file(cwas_workspace, cwas_config)


@pytest.fixture
def create_incomplete_cwas_config_file(cwas_workspace, cwas_config):
    _unset_required_config(cwas_config)
    _create_cwas_config_file(cwas_workspace, cwas_config)


@pytest.fixture
def create_cwas_config_file_without_optional(cwas_workspace, cwas_config):
    _unset_optional_config(cwas_config)
    _create_cwas_config_file(cwas_workspace, cwas_config)


@pytest.fixture
def create_cwas_config_file_invalid_file_path(
    cwas_workspace, cwas_config, invalid_file_path
):
    _set_invalid_file_path(cwas_config, invalid_file_path)
    _create_cwas_config_file(cwas_workspace, cwas_config)


@pytest.fixture
def create_cwas_config_file_invalid_dir_path(
    cwas_workspace, cwas_config, invalid_file_path
):
    _set_invalid_dir_path(cwas_config, invalid_file_path)
    _create_cwas_config_file(cwas_workspace, cwas_config)


@pytest.fixture
def create_cwas_config_file_invalid_vep_path(
    cwas_workspace, cwas_config, invalid_file_path
):
    _set_invalid_vep_path(cwas_config, invalid_file_path)
    _create_cwas_config_file(cwas_workspace, cwas_config)


def _create_cwas_config_file(cwas_workspace, cwas_config):
    config_path = cwas_workspace / "configuration.txt"

    with config_path.open("w") as config_file:
        for k, v in cwas_config.items():
            print(f"{k}={str(v)}", file=config_file)


def _unset_required_config(cwas_config):
    random_config_key = random.choice(
        ["GENE_MATRIX", "ANNOTATION_DATA_DIR", "VEP"]
    )
    cwas_config[random_config_key] = ""


def _unset_optional_config(cwas_config):
    random_config_key = random.choice(
        ["ANNOTATION_KEY_CONFIG", "BIGWIG_CUTOFF_CONFIG"]
    )
    cwas_config[random_config_key] = ""


def _set_invalid_file_path(cwas_config, invalid_file_path):
    random_file_key = random.choice(
        ["GENE_MATRIX", "ANNOTATION_KEY_CONFIG", "BIGWIG_CUTOFF_CONFIG"]
    )
    cwas_config[random_file_key] = invalid_file_path


def _set_invalid_dir_path(cwas_config, invalid_dir_path):
    cwas_config["ANNOTATION_DATA_DIR"] = invalid_dir_path


def _set_invalid_vep_path(cwas_config, invalid_vep_path):
    cwas_config["VEP"] = invalid_vep_path


@pytest.fixture
def configuration_inst():
    inst = Configuration.get_instance()
    return inst


def test_run_configuration_with_incomplete(
    configuration_inst, create_incomplete_cwas_config_file,
):
    with pytest.raises(ValueError):
        configuration_inst.run()


def test_run_configuration(
    cwas_workspace, configuration_inst, create_cwas_config_file
):
    configuration_inst.run()
    _check_config_outputs(cwas_workspace)


def test_run_configuration_without_optional(
    cwas_workspace, configuration_inst, create_cwas_config_file_without_optional
):
    configuration_inst.run()
    _check_config_outputs(cwas_workspace)


def _check_config_outputs(cwas_workspace):
    data_dir_symlink = cwas_workspace / "annotation-data"
    gene_matrix_symlink = cwas_workspace / "gene_matrix.txt"
    bed_key_list = cwas_workspace / "annotation_key_bed.yaml"
    bw_key_list = cwas_workspace / "annotation_key_bw.yaml"
    bw_cutoff_list = cwas_workspace / "annotation_cutoff_bw.yaml"
    category_domain_list = cwas_workspace / "category_domain.yaml"
    redundant_category_table = cwas_workspace / "redundant_category.txt"

    assert data_dir_symlink.is_dir() and data_dir_symlink.is_symlink()
    assert gene_matrix_symlink.is_file() and gene_matrix_symlink.is_symlink()
    assert bed_key_list.is_file()
    assert bw_key_list.is_file()
    assert bw_cutoff_list.is_file()
    assert category_domain_list.is_file()
    assert redundant_category_table.is_file()

    # Teardown
    data_dir_symlink.unlink()
    gene_matrix_symlink.unlink()
    bed_key_list.unlink()
    bw_key_list.unlink()
    bw_cutoff_list.unlink()
    category_domain_list.unlink()
    redundant_category_table.unlink()


def test_run_configuration_with_incomplete(
    configuration_inst, create_incomplete_cwas_config_file,
):
    with pytest.raises(ValueError):
        configuration_inst.run()


def test_run_configuration_with_invalid_file_path(
    configuration_inst, create_cwas_config_file_invalid_file_path
):
    with pytest.raises(FileNotFoundError):
        configuration_inst.run()


def test_run_configuration_with_invalid_dir_path(
    configuration_inst, create_cwas_config_file_invalid_dir_path
):
    with pytest.raises(NotADirectoryError):
        configuration_inst.run()


def test_run_configuration_with_invalid_vep_path(
    configuration_inst, create_cwas_config_file_invalid_vep_path
):
    with pytest.raises(ValueError):
        configuration_inst.run()


def test_env_after_run_configuration(
    configuration_inst, create_cwas_config_file
):
    configuration_inst.run()

    env_keys = [
        "VEP",
        "CWAS_WORKSPACE",
        "ANNOTATION_DATA",
        "GENE_MATRIX",
        "ANNOTATION_BED_KEY",
        "ANNOTATION_BW_KEY",
        "ANNOTATION_BW_CUTOFF",
        "CATEGORY_DOMAIN",
        "REDUNDANT_CATEGORY",
    ]
    for env_key in env_keys:
        assert configuration_inst.get_env(env_key)


def test_get_inst_without_env():
    _make_env_empty()
    with pytest.raises(RuntimeError):
        Configuration.get_instance()


def _make_env_empty():
    env = Env()
    env.reset()
    env.save()
