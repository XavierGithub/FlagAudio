import pytest
import torch
import torchaudio

import flag_gems
from flag_gems.ops.spectrogram import spectrogram as gems_spectrogram

from . import accuracy_utils as utils


def _torch_spectrogram(waveform, n_fft, hop_length, win_length, power, normalized, center, onesided):
    """Reference implementation using torchaudio.functional.spectrogram."""
    window = torch.hann_window(win_length, device=waveform.device)
    return torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        center=center,
        pad_mode="reflect",
        onesided=onesided,
    )


@pytest.mark.spectrogram
@pytest.mark.parametrize("n_fft", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("power", [1.0, 2.0])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_spectrogram_basic(n_fft, power, center, onesided):
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(2, 16000, device=flag_gems.device, dtype=torch.float32)

    ref_out = _torch_spectrogram(waveform, n_fft, hop_length, win_length, power, False, center, onesided)

    window = torch.hann_window(win_length, device=flag_gems.device)
    res_out = gems_spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=False,
        center=center,
        pad_mode="reflect",
        onesided=onesided,
    )

    assert res_out.shape == ref_out.shape, f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"
    # n_fft=1024 has slightly higher floating-point accumulation error
    atol = 1e-3 if n_fft >= 512 else 1e-4
    utils.gems_assert_close(res_out, ref_out, torch.float32, atol=atol)


@pytest.mark.spectrogram
@pytest.mark.parametrize("n_fft", [256, 512])
@pytest.mark.parametrize("power", [None, 1.0, 2.0])
def test_spectrogram_power(n_fft, power):
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(2, 8000, device=flag_gems.device, dtype=torch.float32)

    ref_out = _torch_spectrogram(waveform, n_fft, hop_length, win_length, power, False, True, True)

    window = torch.hann_window(win_length, device=flag_gems.device)
    res_out = gems_spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    assert res_out.shape == ref_out.shape, f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"
    if power is None:
        utils.gems_assert_close(res_out, ref_out, torch.complex64)
    else:
        utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.spectrogram
@pytest.mark.parametrize("n_fft", [256, 512])
@pytest.mark.parametrize("normalized", [True, False, "frame_length", "window"])
def test_spectrogram_normalized(n_fft, normalized):
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(2, 8000, device=flag_gems.device, dtype=torch.float32)

    ref_out = _torch_spectrogram(waveform, n_fft, hop_length, win_length, 2.0, normalized, True, True)

    window = torch.hann_window(win_length, device=flag_gems.device)
    res_out = gems_spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,
        normalized=normalized,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    assert res_out.shape == ref_out.shape, f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"
    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.spectrogram
@pytest.mark.parametrize("n_fft", [256, 512])
@pytest.mark.parametrize("hop_length_factor", [1, 2, 4])
def test_spectrogram_hop_length(n_fft, hop_length_factor):
    hop_length = n_fft // hop_length_factor
    win_length = n_fft
    waveform = torch.randn(1, 8000, device=flag_gems.device, dtype=torch.float32)

    ref_out = _torch_spectrogram(waveform, n_fft, hop_length, win_length, 2.0, False, True, True)

    window = torch.hann_window(win_length, device=flag_gems.device)
    res_out = gems_spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    assert res_out.shape == ref_out.shape, f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"
    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.spectrogram
@pytest.mark.parametrize("n_fft", [256, 512])
def test_spectrogram_win_length_less_than_n_fft(n_fft):
    hop_length = n_fft // 4
    win_length = n_fft // 2
    waveform = torch.randn(1, 8000, device=flag_gems.device, dtype=torch.float32)

    ref_window = torch.hann_window(win_length, device=flag_gems.device)
    ref_out = torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=ref_window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    res_out = gems_spectrogram(
        waveform=waveform,
        pad=0,
        window=ref_window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    assert res_out.shape == ref_out.shape, f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"
    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.spectrogram
@pytest.mark.parametrize("batch_shape", [(8000,), (3, 8000), (2, 3, 8000)])
def test_spectrogram_batch(batch_shape):
    n_fft = 256
    hop_length = n_fft // 4
    win_length = n_fft
    waveform = torch.randn(*batch_shape, device=flag_gems.device, dtype=torch.float32)

    ref_out = _torch_spectrogram(waveform, n_fft, hop_length, win_length, 2.0, False, True, True)

    window = torch.hann_window(win_length, device=flag_gems.device)
    res_out = gems_spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    assert res_out.shape == ref_out.shape, f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"
    utils.gems_assert_close(res_out, ref_out, torch.float32)
