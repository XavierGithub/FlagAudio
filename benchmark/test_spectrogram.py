import pytest
import torch
import torchaudio

import flag_audio
from flag_audio.ops.spectrogram import spectrogram as gems_spectrogram

from .attri_util import DEFAULT_METRICS
from .performance_utils import Benchmark


class SpectrogramBenchmark(Benchmark):
    """Benchmark for spectrogram operation."""

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_fft_values = [256, 512, 1024]
        self.batch_sizes = [1, 4, 16]
        self.hop_factors = [4]  # hop_length = n_fft // hop_factor

    def get_input_iter(self, dtype):
        for n_fft in self.n_fft_values:
            for batch in self.batch_sizes:
                for hop_factor in self.hop_factors:
                    hop_length = n_fft // hop_factor
                    win_length = n_fft
                    time_len = hop_length * 100 + n_fft
                    waveform = torch.randn(
                        batch, time_len, device=self.device, dtype=torch.float32
                    )
                    window = torch.hann_window(
                        win_length, device=self.device, dtype=torch.float32
                    )
                    yield (
                        waveform,
                        window,
                        n_fft,
                        hop_length,
                        win_length,
                    )

    def get_tflops(self, op, *args, **kwargs):
        waveform, window, n_fft, hop_length, win_length = args
        batch = waveform.shape[0]
        time_len = waveform.shape[1]
        num_frames = 1 + (time_len - n_fft) // hop_length
        # FFT is O(N log N) per frame
        flops_per_fft = 5 * n_fft * (n_fft.bit_length() - 1)
        return batch * num_frames * flops_per_fft


@pytest.mark.spectrogram
def test_perf_spectrogram():
    def torch_spec(waveform, window, n_fft, hop_length, win_length):
        return torchaudio.functional.spectrogram(
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

    def gems_spec(waveform, window, n_fft, hop_length, win_length):
        return gems_spectrogram(
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

    bench = SpectrogramBenchmark(
        op_name="spectrogram",
        torch_op=torch_spec,
        dtypes=[torch.float32],
    )
    bench.set_gems(gems_spec)
    bench.run()
