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
def test_accuracy_mask_along_axis_iid(shape, dtype,mask_param, mask_value, axis, p, monkeypatch):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    monkeypatch.setattr(torch, "rand", mock_rand)
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
def test_accuracy_mask_along_axis(shape, dtype,mask_param, mask_value, axis, p, monkeypatch):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_audio.device)
    ref_inp1_tensor = to_reference(input_tensor, True)

    monkeypatch.setattr(torch, "rand", mock_rand)
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


@pytest.mark.convolve
@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        ((64,), (8,)),            # 1D：短核
        ((400,), (7,)),           # 信号长、核短
        ((400,), (400,)),         # valid 退化成单点
        ((7,), (400,)),           # x 比 y 短，验 same 取原始 x 长度
        ((2, 400), (1, 64)),      # 共享核 + 多 batch（y 前导维为 1）
        ((3, 2, 16), (3, 2, 5)),  # 逐信号配对（各自带核）
        ((2, 1, 16), (1, 3, 5)),  # 真广播
        ((1,), (5,)),             # n_len=1 退化
        ((32,), (1,)),            # m_len=1 退化
        ((128,), (1,)),           # out_len 恰好落在 BLOCK_OUT 边界
        ((129,), (1,)),           # out_len 跨 BLOCK_OUT 边界
        ((1, 200), (1, 65)),      # M 跨 BLOCK_T 边界
        ((1, 200), (1, 64)),      # M 恰好等于 BLOCK_T
        ((3, 2, 16), (1, 1, 5)),  # ndim>=3 的共享核
    ],
)
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_convolve(x_shape, y_shape, mode, dtype):
    x = torch.randn(x_shape, dtype=dtype, device=flag_audio.device)
    y = torch.randn(y_shape, dtype=dtype, device=flag_audio.device)
    ref_x = to_reference(x, False)
    ref_y = to_reference(y, False)

    ref_out = torchaudio.functional.convolve(ref_x, ref_y, mode)
    res_out = flag_audio.ops.convolve(ref_x, ref_y, mode)

    # 累加顺序与 cuDNN 不同（我们全程 fp32 顺序累加），按 dtype 放宽绝对容差
    atol = {
        torch.float32: 1e-4,
        torch.float16: 5e-3,
        torch.bfloat16: 4e-2,
    }[dtype]
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.convolve
def test_convolve_invalid_shapes():
    x = torch.randn(2, 8, device=flag_audio.device)
    # 两个操作数维度数不一致
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x, torch.randn(8, device=flag_audio.device))
    # 前导维不可广播
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x, torch.randn(3, 8, device=flag_audio.device))
    # 非法 mode
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x, torch.randn(2, 4, device=flag_audio.device), "invalid")


@pytest.mark.convolve
def test_convolve_unsupported_inputs():
    x = torch.randn(2, 8, device=flag_audio.device)
    y = torch.randn(2, 4, device=flag_audio.device)
    # requires_grad：暂不支持 autograd，必须显式报错而非静默丢梯度
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x.clone().requires_grad_(), y)
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x, y.clone().requires_grad_())
    # complex dtype 不支持
    with pytest.raises(ValueError):
        flag_audio.ops.convolve(x.to(torch.complex64), y)


@pytest.mark.convolve
def test_convolve_empty_batch():
    # batch=0：直接返回空结果，不启动 0 维 grid（torchaudio 此时也会失败，无法对拍）
    x = torch.randn(0, 8, device=flag_audio.device)
    y = torch.randn(1, 4, device=flag_audio.device)
    out = flag_audio.ops.convolve(x, y, "full")
    assert out.shape == (0, 11)
    assert out.dtype == torch.float32


@pytest.mark.convolve
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_accuracy_convolve_noncontiguous(mode):
    # 覆盖 wrapper 里的 .contiguous() 归一化路径：切片、转置、非连续核
    y = torch.randn(1, 8, dtype=torch.float32, device=flag_audio.device)
    base = torch.randn(2, 800, dtype=torch.float32, device=flag_audio.device)
    x_slice = base[:, ::2]  # stride=(800, 2)
    x_transposed = torch.randn(400, 2, dtype=torch.float32, device=flag_audio.device).t()
    for x in (x_slice, x_transposed):
        assert not x.is_contiguous()
        ref_out = torchaudio.functional.convolve(to_reference(x, False), to_reference(y, False), mode)
        res_out = flag_audio.ops.convolve(to_reference(x, False), to_reference(y, False), mode)
        gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)

    # 非连续的核
    x = torch.randn(2, 400, dtype=torch.float32, device=flag_audio.device)
    y_nc = torch.randn(1, 800, dtype=torch.float32, device=flag_audio.device)[:, ::2]
    assert not y_nc.is_contiguous()
    gems_assert_close(
        flag_audio.ops.convolve(x, y_nc, mode),
        torchaudio.functional.convolve(x, y_nc, mode),
        torch.float32,
        atol=1e-4,
    )


@pytest.mark.convolve
def test_accuracy_convolve_large_batch():
    # 回归：batch 若落在 gridDim.y 会撞上 65535 上限而启动失败
    x = torch.randn(70000, 4, dtype=torch.float32, device=flag_audio.device)
    y = torch.randn(1, 4, dtype=torch.float32, device=flag_audio.device)
    ref_out = torchaudio.functional.convolve(x, y, "full")
    res_out = flag_audio.ops.convolve(x, y, "full")
    gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)


@pytest.mark.convolve
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_accuracy_convolve_mode_as_kwarg(mode):
    # benchmark 通过 kwargs 传 mode（字符串无法经 unpack_to_args_kwargs 位置传递）
    x = torch.randn(2, 40, dtype=torch.float32, device=flag_audio.device)
    y = torch.randn(1, 5, dtype=torch.float32, device=flag_audio.device)
    res_out = flag_audio.ops.convolve(x, y, mode=mode)
    ref_out = torchaudio.functional.convolve(x, y, mode)
    gems_assert_close(res_out, ref_out, torch.float32, atol=1e-4)


@pytest.mark.convolve
def test_accuracy_convolve_large_magnitude():
    # 大幅度输入下 fp32 累加的相对误差（相对输出幅度）应保持在 1e-3 以内，
    # 防止将来误改成低精度累加
    scale = 1e4
    x = torch.randn(1, 400, dtype=torch.float32, device=flag_audio.device) * scale
    y = torch.randn(1, 400, dtype=torch.float32, device=flag_audio.device) * scale
    golden = torch.from_numpy(
        np.convolve(
            x.double().cpu().numpy()[0],
            y.double().cpu().numpy()[0],
            mode="full",
        )
    )
    got = flag_audio.ops.convolve(x, y, "full").cpu().double()[0]
    rel_err = (got - golden).abs().max() / golden.abs().max()
    assert rel_err < 1e-3, f"大幅度下相对误差过大: {rel_err:.3e}"
