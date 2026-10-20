# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from typing import Any, Generator, List, Optional, Tuple

import pytest
import torch
import triton
import torchaudio
import sys

import flag_audio

from .performance_utils import Benchmark, GenericBenchmark

class TorchaudioBenchmark(GenericBenchmark):
    """
    benchmark for attention
    """

    def set_more_shapes(self):
        return None


@pytest.mark.gain
@pytest.mark.parametrize("gain_db", [6.0])
def test_perf_gain(gain_db):
    def gain_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, gain_db
    def ref_op(input, gain_db):
        return torchaudio.functional.gain(input, gain_db)

    bench = TorchaudioBenchmark(
        op_name="gain",
        input_fn=gain_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.gain)
    bench.run()

@pytest.mark.DB_to_amplitude
@pytest.mark.parametrize("ref", [3.0])
@pytest.mark.parametrize("power", [0.5])
def test_perf_DB_to_amplitude(ref, power):
    def DB_to_amplitude_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, ref, power
    def ref_op(input, ref, power):
        return torchaudio.functional.DB_to_amplitude(input, ref, power)

    bench = TorchaudioBenchmark(
        op_name="DB_to_amplitude",
        input_fn=DB_to_amplitude_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.DB_to_amplitude)
    bench.run()


class MaskAlongBenchmark(GenericBenchmark):
    """
    benchmark for attention
    """

    def set_more_shapes(self):
        return None

@pytest.mark.mask_along_axis_iid
@pytest.mark.parametrize("mask_param", [3])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [3])
@pytest.mark.parametrize("p", [1])
def test_perf_mask_along_axis_iid(mask_param, mask_value, axis, p):
    def mask_along_axis_iid_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, mask_param, mask_value, axis, p
    def ref_op(input, mask_param, mask_value, axis, p):
        return torchaudio.functional.mask_along_axis_iid(input, mask_param, mask_value, axis, p)

    bench = MaskAlongBenchmark(
        op_name="mask_along_axis_iid",
        input_fn=mask_along_axis_iid_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.mask_along_axis_iid)
    bench.run()


@pytest.mark.mask_along_axis
@pytest.mark.parametrize("mask_param", [6])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [2])
@pytest.mark.parametrize("p", [0.5])
def test_perf_mask_along_axis(mask_param, mask_value, axis, p):
    def mask_along_axis_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, mask_param, mask_value, axis, p
    def ref_op(input, mask_param, mask_value, axis, p):
        return torchaudio.functional.mask_along_axis(input, mask_param, mask_value, axis, p)

    bench = MaskAlongBenchmark(
        op_name="mask_along_axis",
        input_fn=mask_along_axis_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.mask_along_axis)
    bench.run()

@pytest.mark.preemphasis
@pytest.mark.parametrize("coeff", [0.97])
def test_perf_preemphasis(coeff):
    def preemphasis_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, coeff
    def ref_op(input, coeff):
        return torchaudio.functional.preemphasis(input, coeff)

    bench = MaskAlongBenchmark(
        op_name="preemphasis",
        input_fn=preemphasis_input,
        torch_op= ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.preemphasis)
    bench.run()


@pytest.mark.amplitude_to_DB
@pytest.mark.parametrize("multiplier", [3.0])
@pytest.mark.parametrize("amin", [0.5])
@pytest.mark.parametrize("db_multiplier", [1.0])
@pytest.mark.parametrize("top_db", [None, 80.0])
def test_perf_amplitude_to_DB(multiplier, amin, db_multiplier, top_db):
    def amplitude_to_DB_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, multiplier, amin, db_multiplier, top_db
    def ref_op(input, multiplier, amin, db_multiplier, top_db):
        return torchaudio.functional.amplitude_to_DB(input, multiplier, amin, db_multiplier, top_db=top_db)

    bench = MaskAlongBenchmark(
        op_name="amplitude_to_DB",
        input_fn=amplitude_to_DB_input,
        torch_op=ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.amplitude_to_DB)
    bench.run()


@pytest.mark.mu_law_encoding
@pytest.mark.parametrize("quantization_channels", [256])
def test_perf_mu_law_encoding(quantization_channels):
    def mu_law_encoding_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, quantization_channels
    def ref_op(input, quantization_channels):
        return torchaudio.functional.mu_law_encoding(input, quantization_channels)

    bench = TorchaudioBenchmark(
        op_name="mu_law_encoding",
        input_fn=mu_law_encoding_input,
        torch_op=ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.mu_law_encoding)
    bench.run()


@pytest.mark.dcshift
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("shift", [0.5])
@pytest.mark.parametrize("limiter_gain", [None, 0.5])
def test_perf_dcshift(dtype, shift, limiter_gain):
    def dcshift_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        yield input, shift, limiter_gain
    def ref_op(input, shift, limiter_gain):
        return torchaudio.functional.dcshift(input, shift, limiter_gain)

    bench = MaskAlongBenchmark(
        op_name="dcshift",
        input_fn=dcshift_input,
        torch_op=ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.dcshift)
    bench.run()


class SpectralBenchmark(GenericBenchmark):
    """
    benchmark for multiple dimension inputs
    """

    def set_more_shapes(self):
        return None


@pytest.mark.spectral_centroid
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("sample_rate", [16000])
@pytest.mark.parametrize("pad", [0, 1])
@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("hop_length", [160])
@pytest.mark.parametrize("win_length", [400])
def test_perf_spectral_centroid(dtype, sample_rate, pad, n_fft, hop_length, win_length):
    def spectral_centroid_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        windows = torch.hann_window(win_length, device=device)
        yield input, sample_rate, pad, windows, n_fft, hop_length, win_length
    def ref_op(input, sample_rate, pad, windows, n_fft, hop_length, win_length):
        return torchaudio.functional.spectral_centroid(input, sample_rate=sample_rate, pad=pad, window=windows, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    bench = SpectralBenchmark(
        op_name="spectral_centroid",
        input_fn=spectral_centroid_input,
        torch_op=ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.spectral_centroid)
    bench.run()


@pytest.mark.add_noise
@pytest.mark.parametrize("length_factor", [None, 1])
def test_perf_add_noise(length_factor):
    def add_noise_input(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        noise = torch.randn(shape, device=device, dtype=dtype)
        snr = torch.randn(shape[:-1], device=device, dtype=dtype)
        length = torch.ones_like(snr, dtype=torch.int64, device=device) * shape[-1] // 3 if length_factor is not None else None
        yield input, noise, snr, length
    def ref_op(input, noise, snr, length):
        return torchaudio.functional.add_noise(input, noise, snr, lengths=length)

    bench = SpectralBenchmark(
        op_name="add_noise",
        input_fn=add_noise_input,
        torch_op=ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.add_noise)
    bench.run()


@pytest.mark.convolve
@pytest.mark.parametrize("mode", ["full", "same"])
def test_perf_convolve(mode):
    def convolve_input(shape, dtype, device):
        b, n, m = shape
        x = torch.randn(b, n, device=device, dtype=dtype)
        # 共享核：y 前导维为 1，等价于所有信号复用同一条冲激响应
        y = torch.randn(1, m, device=device, dtype=dtype)
        # mode 是字符串，必须走 kwargs（unpack_to_args_kwargs 只透传张量/数字/None）
        yield x, y, {"mode": mode}

    def ref_op(x, y, mode):
        return torchaudio.functional.convolve(x, y, mode)

    bench = SpectralBenchmark(
        op_name="convolve",
        input_fn=convolve_input,
        torch_op=ref_op,
        dtypes=[
            torch.float32,
        ],
    )
    bench.set_gems(flag_audio.ops.convolve)
    bench.run()
