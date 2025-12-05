import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq


def moving_average(x, window_size: int):
    """
    Menghitung rata-rata bergerak (sliding window) dari sinyal 1D.
    Digunakan untuk detrending (menghilangkan tren pelan).
    """
    x = np.asarray(x, dtype=np.float32)

    if window_size < 1:
        raise ValueError("window_size harus >= 1")

    # Jika sinyal terlalu pendek, kecilkan window
    if x.size < window_size:
        window_size = x.size

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    # 'same' -> output panjangnya sama dengan input
    return np.convolve(x, kernel, mode="same")


def detrend_signal(x, window_size: int = 75):
    """
    Menghilangkan tren pelan dari sinyal dengan cara mengurangi
    moving average (sliding window).
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    trend = moving_average(x, window_size=window_size)
    return x - trend


def bandpass_filter(x, fs: float,
                    low: float = 0.67, high: float = 4.0,
                    order: int = 4):
    """
    Menerapkan Butterworth band-pass filter ke sinyal.

    Parameters:
    - x   : sinyal 1D
    - fs  : sampling rate (Hz)
    - low : frekuensi cut-off bawah (Hz)
    - high: frekuensi cut-off atas (Hz)
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq

    if high_n >= 1.0:
        high_n = 0.999  # jaga-jaga biar tidak error

    b, a = butter(order, [low_n, high_n], btype="bandpass")
    y = filtfilt(b, a, x)
    return y


def get_bpm_from_fft(x,
                     fs: float,
                     min_bpm: float = 40.0,
                     max_bpm: float = 240.0):
    """
    Menghitung BPM dari sinyal rPPG dengan FFT.

    Returns:
    - bpm_peak : nilai BPM dominan (float) atau NaN jika gagal
    - bpm_axis : array BPM untuk sumbu x spektrum
    - spec_roi : magnitude spektrum (sesuai bpm_axis)
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.size

    if n < 2:
        return np.nan, None, None

    # FFT hanya bagian frekuensi positif
    yf = rfft(x)
    xf = rfftfreq(n, d=1.0 / fs)  # frekuensi dalam Hz

    bpm = xf * 60.0  # konversi Hz -> BPM
    spectrum = np.abs(yf)

    # Batasi ke rentang detak jantung manusia yang wajar
    mask = (bpm >= min_bpm) & (bpm <= max_bpm)
    if not np.any(mask):
        return np.nan, bpm, spectrum

    bpm_roi = bpm[mask]
    spec_roi = spectrum[mask]

    idx_peak = int(np.argmax(spec_roi))
    bpm_peak = float(bpm_roi[idx_peak])

    return bpm_peak, bpm_roi, spec_roi
