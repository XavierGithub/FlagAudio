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

import pytest
import torch
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas

import flag_audio
import torchaudio

from .accuracy_utils import ASUM_SHAPES, SCALARS, gems_assert_close, to_reference

@pytest.mark.gain
@pytest.mark.parametrize("shape", [(3)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("gain_db", [0.0, 1.0, -6.0, 6.0])
def test_accuracy_gain(shape, gain_db, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.gain(ref_inp1_tensor, gain_db)

    res_out = flag_audio.ops.gain(input_tensor, gain_db)

    gems_assert_close(res_out, ref_out, dtype)



@pytest.mark.DB_to_amplitude
@pytest.mark.parametrize("shape", [(3)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("ref", [3.0])
@pytest.mark.parametrize("power", [0.5])
def test_accuracy_DB_to_amplitude(shape, dtype, ref, power):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.DB_to_amplitude(ref_inp1_tensor, ref, power)
    res_out = flag_audio.ops.DB_to_amplitude(input_tensor, ref, power)

    gems_assert_close(res_out, ref_out, dtype)


def mock_rand(size, **kwargs):
    return torch.ones(size, **kwargs) * 0.6
@pytest.mark.mask_along_axis_iid
@pytest.mark.parametrize("shape", [(2,2,9,4)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("mask_param", [3])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [3])
@pytest.mark.parametrize("p", [1])
def test_accuracy_mask_along_axis_iid(shape, dtype,mask_param, mask_value, axis, p):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    torch.rand = mock_rand
    ref_out = torchaudio.functional.mask_along_axis_iid(ref_inp1_tensor, mask_param, mask_value, axis, p)
    res_out = flag_audio.ops.mask_along_axis_iid(input_tensor, mask_param, mask_value, axis, p)

    gems_assert_close(res_out, ref_out, dtype)

@pytest.mark.mask_along_axis
@pytest.mark.parametrize("shape", [(2,2,4,4)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("mask_param", [6])
@pytest.mark.parametrize("mask_value", [0.0])
@pytest.mark.parametrize("axis", [2])
@pytest.mark.parametrize("p", [0.5])
def test_accuracy_mask_along_axis(shape, dtype,mask_param, mask_value, axis, p):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    torch.rand = mock_rand
    ref_out = torchaudio.functional.mask_along_axis(ref_inp1_tensor, mask_param, mask_value, axis, p)
    res_out = flag_audio.ops.mask_along_axis(input_tensor, mask_param, mask_value, axis, p)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.preemphasis
@pytest.mark.parametrize("shape", [(2, 2)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("coeff", [0.97])
def test_accuracy_preemphasis(shape, coeff, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.preemphasis(ref_inp1_tensor, coeff)

    res_out = flag_audio.ops.preemphasis(input_tensor, coeff)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mu_law_encoding
@pytest.mark.parametrize("shape", [(3)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("quantization_channels", [256])
def test_accuracy_mu_law_encoding(shape, dtype, quantization_channels):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.mu_law_encoding(ref_inp1_tensor, quantization_channels)

    res_out = flag_audio.ops.mu_law_encoding(input_tensor, quantization_channels)

    gems_assert_close(res_out, ref_out, torch.int64)


@pytest.mark.amplitude_to_DB
@pytest.mark.parametrize("shape", [(5, 2, 2, 2)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("multiplier", [3.0])
@pytest.mark.parametrize("amin", [0.5])
@pytest.mark.parametrize("db_multiplier", [1.0])
@pytest.mark.parametrize("top_db", [None, 80.0])
def test_accuracy_amplitude_to_DB(shape, dtype, multiplier, amin, db_multiplier, top_db):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.amplitude_to_DB(ref_inp1_tensor, multiplier, amin, db_multiplier, top_db=top_db)
    res_out = flag_audio.ops.amplitude_to_DB(input_tensor, multiplier, amin, db_multiplier, top_db=top_db)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.dcshift
@pytest.mark.parametrize("shape", [(2, 2)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("shift", [0.5])
@pytest.mark.parametrize("limiter_gain", [None, 0.5])
def test_accuracy_dcshift(shape, dtype, shift, limiter_gain):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    ref_out = torchaudio.functional.dcshift(ref_inp1_tensor, shift, limiter_gain)
    res_out = flag_audio.ops.dcshift(input_tensor, shift, limiter_gain)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.spectral_centroid
@pytest.mark.parametrize("shape", [(1000,), (2, 1000), (2, 2, 1000)])
@pytest.mark.parametrize("dtype", [(torch.float32)])
@pytest.mark.parametrize("sample_rate", [16000])
@pytest.mark.parametrize("pad", [0, 1])
@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("hop_length", [160])
@pytest.mark.parametrize("win_length", [400])
def test_accuracy_spectral_centroid(shape, dtype, sample_rate, pad, n_fft, hop_length, win_length):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    windows = torch.hann_window(win_length).to(flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, False)
    ref_win_tensor = to_reference(windows, False)

    ref_out = torchaudio.functional.spectral_centroid(ref_inp1_tensor, sample_rate, pad, ref_win_tensor, n_fft, hop_length, win_length)
    res_out = flag_audio.ops.spectral_centroid(input_tensor, sample_rate, pad, ref_win_tensor, n_fft, hop_length, win_length)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add_noise
@pytest.mark.parametrize(
    "shape",
    [
        (8,),          # 1D：单条 + 标量 snr
        (2, 8),        # 2D：多条
        (256,),        # L 恰好 = BLOCK_SIZE，边界
        (257,),        # L = BLOCK_SIZE+1，第二段只有 1 个点
        (3, 2, 1000),  # 多维 + 非整除多段
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length_ratio", [None, 1.0, 0.5])
def test_accuracy_add_noise(shape, dtype, length_ratio):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    noise = torch.rand(shape, dtype=dtype, device=flag_audio.device)
    snr = torch.rand(shape[:-1], dtype=dtype, device=flag_audio.device)
    length = (
        torch.ones_like(snr, dtype=torch.int64, device=flag_audio.device)
        * int(shape[-1] * length_ratio)
        if length_ratio is not None
        else None
    )
    ref_inp1_tensor = to_reference(input_tensor, False)
    ref_noise_tensor = to_reference(noise, False)
    ref_snr_tensor = to_reference(snr, False)
    ref_length_tensor = to_reference(length, False) if length_ratio is not None else None

    ref_out = torchaudio.functional.add_noise(ref_inp1_tensor, ref_noise_tensor, ref_snr_tensor, lengths=ref_length_tensor)
    res_out = flag_audio.ops.add_noise(ref_inp1_tensor, ref_noise_tensor, ref_snr_tensor, lengths=ref_length_tensor)

    # half 精度下 scale 的舍入路径与 torchaudio 不同（我们全程 fp32），
    # 差异实测仅 1~3 ULP，按 dtype 放宽绝对容差
    atol = {
        torch.float32: 1e-4,
        torch.float16: 5e-3,
        torch.bfloat16: 4e-2,
    }[dtype]
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.add_noise
def test_add_noise_invalid_shapes():
    w = torch.randn(2, 8, device=flag_audio.device)
    n = torch.rand(2, 8, device=flag_audio.device)
    # snr 维度数不匹配：应为 (2,) 而不是标量
    with pytest.raises(ValueError):
        flag_audio.ops.add_noise(w, n, torch.tensor(0.5, device=flag_audio.device))
    # 最后一维长度不一致
    with pytest.raises(ValueError):
        flag_audio.ops.add_noise(
            w,
            torch.rand(2, 7, device=flag_audio.device),
            torch.rand(2, device=flag_audio.device),
        )
