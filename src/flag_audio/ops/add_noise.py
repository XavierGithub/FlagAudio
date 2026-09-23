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

import torch
import triton
import triton.language as tl


@triton.jit
def partial_energy_kernel(
    waveform_ptr,  # (batch, L)
    noise_ptr,  # (batch, L)
    lengths_ptr,  # (batch,) or None
    partial_w_ptr,  # (batch, T)
    partial_n_ptr,  # (batch, T)
    L,
    T,
    has_lengths: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 一维 grid 共 B*T 个 block，自己解出"第几条音频、第几段"
    pid = tl.program_id(0)
    b = pid // T
    tile = pid % T

    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 两层 mask：段边界 + lengths 有效区
    mask = offsets < L
    if has_lengths:
        length = tl.load(lengths_ptr + b)
        mask = mask & (offsets < length)

    w = tl.load(waveform_ptr + b * L + offsets, mask=mask, other=0.0)
    n = tl.load(noise_ptr + b * L + offsets, mask=mask, other=0.0)
    # block 内先做一次树形归约，每个 block 只输出两个标量
    tl.store(partial_w_ptr + b * T + tile, tl.sum(w * w))
    tl.store(partial_n_ptr + b * T + tile, tl.sum(n * n))


@triton.jit
def finalize_scale_kernel(
    partial_w_ptr,  # (batch, T)
    partial_n_ptr,  # (batch, T)
    snr_ptr,  # (batch,)
    scale_ptr,  # (batch,)
    T,
):
    b = tl.program_id(0)
    energy_w = 0.0
    energy_n = 0.0
    # 只剩标量累加，循环再长也很便宜
    for tile in range(T):
        energy_w += tl.load(partial_w_ptr + b * T + tile)
        energy_n += tl.load(partial_n_ptr + b * T + tile)

    original_snr_db = (tl.log(energy_w) - tl.log(energy_n)) / tl.log(10.0) * 10
    scale = tl.exp(
        (original_snr_db - tl.load(snr_ptr + b)) / 20 * tl.log(10.0)
    )
    tl.store(scale_ptr + b, scale)


@triton.jit
def apply_noise_kernel(
    waveform_ptr,  # (batch, L)
    noise_ptr,  # (batch, L)
    scale_ptr,  # (batch,)
    output_ptr,  # (batch, L)
    L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    scale = tl.load(scale_ptr + b)
    block_start = b * L + pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (b + 1) * L
    w = tl.load(waveform_ptr + offsets, mask=mask, other=0.0)
    n = tl.load(noise_ptr + offsets, mask=mask, other=0.0)
    result = w + n * scale
    tl.store(output_ptr + offsets, result, mask=mask)


def add_noise(
    waveform: torch.Tensor,
    noise: torch.Tensor,
    snr: torch.Tensor,
    lengths: torch.Tensor = None,
) -> torch.Tensor:

    if not (
        waveform.ndim - 1 == noise.ndim - 1 == snr.ndim
        and (lengths is None or lengths.ndim == snr.ndim)
    ):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(
            f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)})."
        )
    # 展平前导维度：不管原来几维，统一看成 (B, L)
    total_elements = waveform.numel()  # 张量里元素总数
    batch_size = total_elements // L  # 总数 ÷ 每条长度 = 条数
    # reshape 保证内存连续，kernel 里才能用 b*L+offset 直接寻址
    waveform_2d = waveform.reshape(batch_size, L)
    noise_2d = noise.reshape(batch_size, L)
    snr_flat = snr.reshape(batch_size)
    lengths_flat = lengths.reshape(batch_size) if lengths is not None else None

    BLOCK_SIZE = 256
    T = triton.cdiv(L, BLOCK_SIZE)  # 一条音频被切成多少段
    has_lengths = lengths is not None

    # 中间缓冲区：每条音频、每段各一个部分和
    partial_w = torch.empty(batch_size, T, device=waveform.device)
    partial_n = torch.empty(batch_size, T, device=waveform.device)

    # 阶段1：B*T 个 block，在 batch 维和 L 维上同时并行
    partial_energy_kernel[(batch_size * T,)](
        waveform_2d,
        noise_2d,
        lengths_flat,
        partial_w,
        partial_n,
        L,
        T,
        has_lengths,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 阶段2：B 个 block 汇总部分和，算最终 scale
    scale = torch.empty(batch_size, device=waveform.device)
    finalize_scale_kernel[(batch_size,)](
        partial_w,
        partial_n,
        snr_flat,
        scale,
        T,
    )

    # 分配输出
    output = torch.empty_like(waveform_2d)
    grid = (T, batch_size)
    apply_noise_kernel[grid](
        waveform_2d,
        noise_2d,
        scale,
        output,
        L,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(waveform.shape)


def test_op():
    import torchaudio

    x = torch.rand([5, 2, 2, 8], dtype=torch.float32).to("cuda")
    noise = torch.rand(x.shape, dtype=torch.float32).to("cuda")
    snr = torch.rand(x.shape[:-1], dtype=torch.float32).to("cuda")

    output_tensor = add_noise(x, noise, snr)
    golden_tensor = torchaudio.functional.add_noise(x, noise, snr)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)
    
    length = torch.ones_like(snr, dtype=torch.int64).to("cuda") * (x.shape[-1] // 2)
    output_tensor = add_noise(x, noise, snr, lengths=length)
    golden_tensor = torchaudio.functional.add_noise(x, noise, snr, lengths=length)
    torch.testing.assert_close(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    test_op()
