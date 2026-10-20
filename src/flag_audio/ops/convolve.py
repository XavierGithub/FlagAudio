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
import torchaudio
import triton
import triton.language as tl


@triton.jit
def convolve_kernel(
    x_ptr,  # (B, N) 已归一化为连续
    y_ptr,  # (B, M) 或共享核时 (M,)
    out_ptr,  # (B, out_len)
    out_len,
    n_len,
    m_len,
    start,  # 当前 mode 在 full 卷积域中的起点
    x_row_stride,
    y_row_stride,  # 共享核时为 0
    BLOCK_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # batch 放在 grid 第 0 维：gridDim.x 上限 2^31-1，而 gridDim.y 上限仅 65535，
    # 信号条数很容易超过 65535（小 batch 维度的音频数据集），放错轴会直接启动失败
    pid_b = tl.program_id(0)  # 信号序号
    pid_o = tl.program_id(1)  # 输出分块
    x_row = pid_b * x_row_stride
    y_row = pid_b * y_row_stride

    o = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    omask = o < out_len
    # 映射回 full 卷积域坐标：out[o] 即 full 卷积的第 start+o 个点
    j = start + o

    # fp32 累加，低精度输入在乘之前统一升精度
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)
    for t0 in range(0, m_len, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        tmask = t < m_len
        yv = tl.load(y_ptr + y_row + t, mask=tmask, other=0.0).to(tl.float32)
        # out[j] = Σ_m x[j-m] · y[m]，与 torchaudio 同向（无需翻转核）
        xi = j[:, None] - t[None, :]
        xmask = omask[:, None] & tmask[None, :] & (xi >= 0) & (xi < n_len)
        xv = tl.load(x_ptr + x_row + xi, mask=xmask, other=0.0).to(tl.float32)
        # 寄存器内沿 tap 维归约，一个 block 只输出一段连续结果
        acc += tl.sum(xv * yv[None, :], axis=1)

    tl.store(
        out_ptr + pid_b * out_len + o,
        acc.to(out_ptr.dtype.element_ty),
        mask=omask,
    )


def _check_shape_compatible(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.ndim != y.ndim:
        raise ValueError(
            f"The operands must be the same dimension (got {x.ndim} and {y.ndim})."
        )
    for i in range(x.ndim - 1):
        xi, yi = x.size(i), y.size(i)
        if xi == yi or xi == 1 or yi == 1:
            continue
        raise ValueError(
            f"Leading dimensions of x and y are not broadcastable (got {x.shape} and {y.shape})."
        )


def _check_convolve_mode(mode: str) -> None:
    if mode not in ("full", "valid", "same"):
        raise ValueError(
            f"Unrecognized mode value '{mode}'. Please specify one of ['full', 'valid', 'same']."
        )


def _check_dtype_supported(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.is_complex() or y.is_complex():
        raise ValueError(
            f"convolve does not support complex dtypes (got {x.dtype} and {y.dtype})."
        )
    if x.requires_grad or y.requires_grad:
        raise ValueError(
            "convolve does not support autograd yet; detach the inputs before calling "
            "to avoid silently losing gradients."
        )


def convolve(x: torch.Tensor, y: torch.Tensor, mode: str = "full") -> torch.Tensor:
    """沿最后一维做真卷积（与 torchaudio.functional.convolve 对齐）。

    单个 Triton kernel 完成计算：不做 broadcast/flip/reshape 物化、不走 cuDNN
    分组卷积，并且只计算 mode 需要的输出区间。

    注意：暂不支持 autograd（torchaudio 可微），requires_grad 的输入会报错而非
    静默丢失梯度；complex dtype 亦不支持。
    """
    _check_shape_compatible(x, y)
    _check_convolve_mode(mode)
    _check_dtype_supported(x, y)

    n_len = x.size(-1)
    m_len = y.size(-1)
    full_len = n_len + m_len - 1

    # 卷积可交换，因此不需要像 torchaudio 那样交换 x/y；注意 same 取的是原始 x 的长度
    if mode == "full":
        out_len, start = full_len, 0
    elif mode == "valid":
        out_len = abs(n_len - m_len) + 1
        start = min(n_len, m_len) - 1
    else:  # same
        out_len = n_len
        start = (full_len - n_len) // 2

    # 前导维广播：各维相等或为 1，展开后再展平成 (B, len)
    lead = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    batch = 1
    for d in lead:
        batch *= d

    if all(s == 1 for s in y.shape[:-1]):
        # 共享核（最常见的 y 为 1D 情形）：跨信号复用同一个核，行步长置 0
        x2 = x.expand(*lead, n_len).reshape(batch, n_len).contiguous()
        y2 = y.reshape(m_len).contiguous()
        x_row_stride, y_row_stride = n_len, 0
    else:
        x2 = x.expand(*lead, n_len).reshape(batch, n_len).contiguous()
        y2 = y.expand(*lead, m_len).reshape(batch, m_len).contiguous()
        x_row_stride, y_row_stride = n_len, m_len

    # 非浮点输入升为 fp32 输出（torchaudio 对非浮点输入直接报错，此处为扩展行为）
    out_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    out = torch.empty(tuple(lead) + (out_len,), dtype=out_dtype, device=x.device)
    if batch == 0:
        # 空 batch：Triton 不接受 0 维 grid，直接返回空结果（torchaudio 此时也会失败）
        return out
    out2d = out.view(batch, out_len)

    BLOCK_OUT, BLOCK_T = 128, 64
    # 第 0 维放 batch（见 kernel 内注释），第 1 维放输出分块
    grid = (batch, triton.cdiv(out_len, BLOCK_OUT))
    convolve_kernel[grid](
        x2,
        y2,
        out2d,
        out_len,
        n_len,
        m_len,
        start,
        x_row_stride,
        y_row_stride,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )

    return out


def test_op():
    for x_shape, y_shape in [
        ((8,), (3,)),
        ((400,), (7,)),
        ((7,), (400,)),
        ((2, 16), (1, 5)),  # 共享核：y 前导维为 1
    ]:
        x = torch.randn(x_shape, dtype=torch.float32, device="cuda")
        y = torch.randn(y_shape, dtype=torch.float32, device="cuda")
        for mode in ("full", "valid", "same"):
            out = convolve(x, y, mode)
            golden = torchaudio.functional.convolve(x, y, mode)
            torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-5)

    # 逐信号配对（真广播）
    x = torch.randn(2, 1, 16, dtype=torch.float32, device="cuda")
    y = torch.randn(1, 3, 5, dtype=torch.float32, device="cuda")
    for mode in ("full", "valid", "same"):
        torch.testing.assert_close(
            convolve(x, y, mode),
            torchaudio.functional.convolve(x, y, mode),
            rtol=1e-4,
            atol=1e-5,
        )


if __name__ == "__main__":
    test_op()
