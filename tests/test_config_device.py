import argparse

import pytest
import torch

from zahhak_snakehandler.agent import resolve_device
from zahhak_snakehandler.config import Config, add_common_args, config_from_args


def parse(argv):
    p = argparse.ArgumentParser()
    add_common_args(p)
    return p.parse_args(argv)


def test_gpu_preset_is_larger_than_default():
    cpu, gpu = Config(), Config.gpu_preset()
    assert gpu.batch_size > cpu.batch_size
    assert sum(gpu.hidden_sizes) > sum(cpu.hidden_sizes)


def test_cli_overrides():
    cfg = config_from_args(parse(["--preset", "gpu", "--seed", "9", "--hidden", "32", "32", "--device", "cpu"]))
    assert cfg.seed == 9 and cfg.hidden_sizes == (32, 32) and cfg.device == "cpu"
    assert cfg.batch_size == Config.gpu_preset().batch_size


def test_checkpoint_path_uses_model_name():
    assert Config().checkpoint_path("best").endswith("Zahhak-snakehandler_best.pt")


def test_resolve_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present on this machine")
def test_explicit_cuda_fails_loudly_without_cuda():
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
