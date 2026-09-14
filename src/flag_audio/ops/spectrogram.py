import logging
import math
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from flag_gems.ops.fft import _bitrev_indices, _twiddle_tables, _is_power_of_two, _log2
from flag_gems.runtime import torch_device_fn, device as gems_device

logger = logging.getLogger(__name__)


@triton.jit
def _spectrogram_kernel(
    waveform_ptr,
    window_ptr,
    out_ptr,
    bitrev_ptr,
    twiddle_real_ptr,
    twiddle_imag_ptr,
    stride_wave_batch,
    stride_wave_time,
    hop_length,
    num_frames,
    n_fft: tl.constexpr,
    n_freq: tl.constexpr,
    LOG_N: tl.constexpr,
    N_RADIX4: tl.constexpr,
    FIRST_STAGE: tl.constexpr,
    POWER: tl.constexpr,
    ONE_SIDED: tl.constexpr,
    BLOCK_FREQ: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_frames
    frame_idx = pid % num_frames

    offs = tl.arange(0, n_fft)
    frame_start = frame_idx * hop_length

    # 1) Load frame data with window applied
    sample_idx = frame_start + offs
    samples = tl.load(
        waveform_ptr + batch_idx * stride_wave_batch + sample_idx * stride_wave_time,
        mask=offs < n_fft,
        other=0.0,
    ).to(tl.float32)

    win = tl.load(window_ptr + offs, mask=offs < n_fft, other=0.0).to(tl.float32)
    frame = samples * win

    # 2) Bit-reversal permutation (DIT input reordering)
    rev = tl.load(bitrev_ptr + offs, mask=offs < n_fft, other=0)
    real = tl.gather(frame, rev, axis=0)
    imag = tl.zeros_like(real)

    # 3) Radix-2 first stage (only when LOG_N is odd)
    if LOG_N % 2 == 1:
        idx = offs
        even_idx = idx & ~1      # clear bit 0
        odd_idx = even_idx | 1    # set bit 0

        u_real = tl.gather(real, even_idx, axis=0)
        u_imag = tl.gather(imag, even_idx, axis=0)
        v_real = tl.gather(real, odd_idx, axis=0)
        v_imag = tl.gather(imag, odd_idx, axis=0)

        tw_real = tl.load(twiddle_real_ptr + 0)  # twiddle[0] = 1
        tw_imag = tl.load(twiddle_imag_ptr + 0)  # twiddle[0] = 0

        v_tw_real = v_real * tw_real - v_imag * tw_imag
        v_tw_imag = v_real * tw_imag + v_imag * tw_real

        is_even = (idx & 1) == 0
        real = tl.where(is_even, u_real + v_tw_real, u_real - v_tw_real)
        imag = tl.where(is_even, u_imag + v_tw_imag, u_imag - v_tw_imag)

    # 4) Radix-4 stages
    for r4 in tl.static_range(N_RADIX4):
        stage_s = FIRST_STAGE + r4 * 2
        m = 1 << (stage_s + 1)
        quarter = m >> 2
        half = m >> 1

        idx = offs
        pos = idx & (m - 1)
        j = pos & (quarter - 1)
        base = idx - pos
        i0 = base + j
        i1 = i0 + quarter
        i2 = i1 + quarter
        i3 = i2 + quarter

        x0_real = tl.gather(real, i0, axis=0)
        x0_imag = tl.gather(imag, i0, axis=0)
        x1_real = tl.gather(real, i1, axis=0)
        x1_imag = tl.gather(imag, i1, axis=0)
        x2_real = tl.gather(real, i2, axis=0)
        x2_imag = tl.gather(imag, i2, axis=0)
        x3_real = tl.gather(real, i3, axis=0)
        x3_imag = tl.gather(imag, i3, axis=0)

        base_tw1 = (1 << (stage_s - 1)) - 1
        base_tw2 = (1 << stage_s) - 1
        tw1_idx = base_tw1 + j
        tw2_idx = base_tw2 + j
        tw1_real = tl.load(twiddle_real_ptr + tw1_idx, mask=offs < n_fft, other=1.0)
        tw1_imag = tl.load(twiddle_imag_ptr + tw1_idx, mask=offs < n_fft, other=0.0)
        tw2_real = tl.load(twiddle_real_ptr + tw2_idx, mask=offs < n_fft, other=1.0)
        tw2_imag = tl.load(twiddle_imag_ptr + tw2_idx, mask=offs < n_fft, other=0.0)

        # Twiddle multiplication
        t1_real = x1_real * tw1_real - x1_imag * tw1_imag
        t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
        t3_real = x3_real * tw1_real - x3_imag * tw1_imag
        t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

        # Butterfly
        u0_real = x0_real + t1_real
        u0_imag = x0_imag + t1_imag
        u1_real = x0_real - t1_real
        u1_imag = x0_imag - t1_imag
        v0_real = x2_real + t3_real
        v0_imag = x2_imag + t3_imag
        v1_real = x2_real - t3_real
        v1_imag = x2_imag - t3_imag

        v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
        v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
        # w3 = tw2 * (-j) = (tw2_imag, -tw2_real)
        v1_tw_real = v1_real * tw2_imag - v1_imag * (-tw2_real)
        v1_tw_imag = v1_real * (-tw2_real) + v1_imag * tw2_imag

        o0_real = u0_real + v0_tw_real
        o0_imag = u0_imag + v0_tw_imag
        o2_real = u0_real - v0_tw_real
        o2_imag = u0_imag - v0_tw_imag
        o1_real = u1_real + v1_tw_real
        o1_imag = u1_imag + v1_tw_imag
        o3_real = u1_real - v1_tw_real
        o3_imag = u1_imag - v1_tw_imag

        three_quarter = quarter + half
        m0 = pos < quarter
        m1 = (pos >= quarter) & (pos < half)
        m2 = (pos >= half) & (pos < three_quarter)
        real = tl.where(m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real)))
        imag = tl.where(m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag)))

    # 5) Extract output frequencies and compute magnitude/power
    if ONE_SIDED:
        freq_offs = tl.arange(0, BLOCK_FREQ)
        freq_mask = freq_offs < n_freq
        f_real = tl.gather(real, freq_offs, axis=0)
        f_imag = tl.gather(imag, freq_offs, axis=0)
    else:
        freq_offs = offs
        freq_mask = offs < n_fft
        f_real = real
        f_imag = imag

    if POWER < 0:
        # Complex output: store real and imag interleaved
        out_stride = n_freq * 2
        tl.store(out_ptr + pid * out_stride + freq_offs * 2, f_real, mask=freq_mask)
        tl.store(out_ptr + pid * out_stride + freq_offs * 2 + 1, f_imag, mask=freq_mask)
    else:
        mag = tl.sqrt(f_real * f_real + f_imag * f_imag)
        if POWER == 2.0:
            mag = mag * mag
        elif POWER != 1.0:
            mag = tl.math.pow(mag, POWER)
        tl.store(out_ptr + pid * n_freq + freq_offs, mag, mask=freq_mask)


def spectrogram(
    waveform: torch.Tensor,
    pad: int = 0,
    window: Optional[torch.Tensor] = None,
    n_fft: int = 400,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    power: Optional[float] = 2.0,
    normalized: Union[bool, str] = False,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
) -> torch.Tensor:
    """Compute a spectrogram from a waveform using Triton.

    Args:
        waveform: Input waveform of shape (..., time).
        pad: Double-sided padding applied before STFT.
        window: Window tensor. If None, a rectangular window is used.
        n_fft: FFT size (must be power of two, <= 1024).
        hop_length: Hop length between frames. Default: n_fft // 4.
        win_length: Window length. Default: n_fft.
        power: Exponent for magnitude. None = complex, 1.0 = magnitude, 2.0 = power.
        normalized: False, True/"window", or "frame_length".
        center: If True, pad waveform so frame t is centered at t * hop_length.
        pad_mode: Padding mode for center (default "reflect").
        onesided: If True, return n_fft//2 + 1 frequencies.

    Returns:
        Spectrogram tensor of shape (..., freq, frames).
    """
    logger.debug("GEMS SPECTROGRAM")

    if not _is_power_of_two(n_fft) or n_fft > 1024:
        raise ValueError(
            f"n_fft={n_fft} must be a power of two and <= 1024"
        )

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    device = waveform.device
    dtype = waveform.dtype

    # Step 1: pad
    if pad > 0:
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    # Step 2: center padding (ensure 2D for reflect mode)
    if center:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            was_1d = True
        else:
            was_1d = False
        waveform = torch.nn.functional.pad(
            waveform, (n_fft // 2, n_fft // 2), pad_mode
        )
        if was_1d:
            waveform = waveform.squeeze(0)

    # Step 3: window
    if window is None:
        window = torch.ones(win_length, device=device, dtype=torch.float32)
    else:
        window = window.to(torch.float32)

    # Save original window for normalization (before padding)
    orig_window = window

    # Pad window to n_fft (centered, matching torch.stft behavior)
    if win_length < n_fft:
        left_pad = (n_fft - win_length) // 2
        right_pad = n_fft - win_length - left_pad
        window = torch.nn.functional.pad(window, (left_pad, right_pad))

    # Step 4: flatten batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    batch = waveform.shape[0]
    time_len = waveform.shape[1]

    # Step 5: compute frame count
    num_frames = 1 + (time_len - n_fft) // hop_length
    n_freq = n_fft // 2 + 1 if onesided else n_fft

    if num_frames <= 0:
        out_shape = shape[:-1] + (n_freq, 0)
        if power is None:
            return torch.empty(out_shape, dtype=torch.complex64, device=device)
        return torch.empty(out_shape, dtype=torch.float32, device=device)

    # Prepare FFT lookup tables
    bitrev = _bitrev_indices(n_fft, device)
    tw_real, tw_imag = _twiddle_tables(n_fft, device)
    log_n = _log2(n_fft)
    if log_n % 2 == 1:
        n_radix4 = (log_n - 1) // 2
        first_stage = 2
    else:
        n_radix4 = log_n // 2
        first_stage = 1

    # Convert to float32 for computation
    waveform_f32 = waveform.to(torch.float32).contiguous()
    window_f32 = window.to(torch.float32).contiguous()

    # Use POWER=-1 as sentinel for complex output
    power_const = -1.0 if power is None else float(power)

    # BLOCK_FREQ must be power of 2 for tl.arange
    block_freq = 1
    while block_freq < n_freq:
        block_freq *= 2

    total_programs = batch * num_frames

    with torch_device_fn.device(device):
        if power is None:
            out = torch.empty(
                (total_programs, n_freq, 2), device=device, dtype=torch.float32
            )
            _spectrogram_kernel[(total_programs,)](
                waveform_f32,
                window_f32,
                out,
                bitrev,
                tw_real,
                tw_imag,
                waveform_f32.stride(0),
                waveform_f32.stride(1),
                hop_length,
                num_frames,
                n_fft=n_fft,
                n_freq=n_freq,
                LOG_N=log_n,
                N_RADIX4=n_radix4,
                FIRST_STAGE=first_stage,
                POWER=power_const,
                ONE_SIDED=onesided,
                BLOCK_FREQ=block_freq,
                num_warps=4,
                num_stages=1,
            )
            spec = torch.complex(out[..., 0], out[..., 1])
            spec = spec.reshape(batch, num_frames, n_freq).permute(0, 2, 1)
            spec = spec.reshape(shape[:-1] + (n_freq, num_frames))
        else:
            out = torch.empty(
                (total_programs, n_freq), device=device, dtype=torch.float32
            )
            _spectrogram_kernel[(total_programs,)](
                waveform_f32,
                window_f32,
                out,
                bitrev,
                tw_real,
                tw_imag,
                waveform_f32.stride(0),
                waveform_f32.stride(1),
                hop_length,
                num_frames,
                n_fft=n_fft,
                n_freq=n_freq,
                LOG_N=log_n,
                N_RADIX4=n_radix4,
                FIRST_STAGE=first_stage,
                POWER=power_const,
                ONE_SIDED=onesided,
                BLOCK_FREQ=block_freq,
                num_warps=4,
                num_stages=1,
            )
            spec = out.reshape(batch, num_frames, n_freq).permute(0, 2, 1)
            spec = spec.reshape(shape[:-1] + (n_freq, num_frames))

    # Step 6: normalization
    # torchaudio normalizes the complex STFT (before abs/pow), so the effective
    # divisor for real output is: (window_energy_sqrt)^power = window_energy^(power/2)
    # For complex output (power=None), it's just window_energy_sqrt.
    if normalized is True or normalized == "window":
        window_energy_sqrt = orig_window.pow(2.0).sum().sqrt()
        if power is not None:
            spec = spec / (window_energy_sqrt ** power)
        else:
            spec = spec / window_energy_sqrt
    elif normalized == "frame_length":
        if power is not None:
            spec = spec * (n_fft ** (-power / 2.0))
        else:
            spec = spec * (n_fft ** -0.5)

    # Cast to original dtype for real outputs
    if power is not None and dtype != torch.float32:
        spec = spec.to(dtype)

    return spec
