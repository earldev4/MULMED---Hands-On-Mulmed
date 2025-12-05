from utils import *
import cv2
import numpy as np
import os
import dlib
import matplotlib.pyplot as plt
import time
from collections import deque
import mediapipe as mp
from utils import detrend_signal, bandpass_filter, get_bpm_from_fft


class RPPGPipeline:
    """
    Pipeline utama rPPG:
    - Buka webcam
    - Deteksi wajah (MediaPipe)
    - Ambil ROI dahi
    - Ekstraksi sinyal hijau (spatial averaging)
    - Sliding window + filtering + FFT -> BPM
    """

    def __init__(
        self,
        camera_index: int = 0,
        window_seconds: float = 10.0,
        fps_guess: float = 30.0,
        min_seconds_for_bpm: float = 5.0,
        show_debug: bool = True,
    ):
        # Parameter input
        self.camera_index = camera_index
        self.window_seconds = window_seconds
        self.fps_guess = fps_guess
        self.min_seconds_for_bpm = min_seconds_for_bpm
        self.show_debug = show_debug

        # Buffer sliding window untuk sinyal dan waktu
        max_len = int(window_seconds * fps_guess)
        self.signal_buffer = deque(maxlen=max_len)  # nilai mean hijau
        self.time_buffer = deque(maxlen=max_len)    # timestamp

        # Estimasi BPM (raw dan smoothed)
        self.current_bpm = np.nan
        self.smooth_bpm = np.nan
        self.alpha_bpm = 0.8  # smoothing faktor (0-1), makin besar -> makin halus

        # Inisialisasi MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )

    # ========================
    # Bagian deteksi wajah / ROI
    # ========================

    def _detect_face_bbox(self, frame_rgb):
        """
        Mendeteksi wajah dan mengembalikan bounding box (x, y, w, h) dalam pixel.
        Jika tidak ada wajah -> None.
        """
        results = self.face_detector.process(frame_rgb)

        if not results.detections:
            return None

        h, w, _ = frame_rgb.shape

        # Ambil deteksi dengan skor tertinggi
        det = max(results.detections, key=lambda d: d.score[0])
        bbox = det.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # --- Tambahan: geser bbox sedikit ke atas dan diperpanjang ---
        shift_up = int(0.10 * bh)   # geser 10% tinggi wajah ke atas
        y = max(0, y - shift_up)
        bh = min(h - y, bh + shift_up)   # tinggi disesuaikan lagi

        # Clamp biar tidak keluar frame
        x = max(0, x)
        y = max(0, y)
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        return x, y, bw, bh

    def _get_forehead_roi(self, frame_bgr, face_bbox):
        """
        Ambil ROI di area jidat/dahi, bukan di mata/alis.
        Diatur supaya area yang diambil adalah sekitar
        seperempat bagian atas wajah.
        """
        x, y, bw, bh = face_bbox

        # Ambil kira-kira 0%–25% bagian atas wajah -> jidat
        top_y = y
        bottom_y = y + int(0.25 * bh)

        # Ambil bagian tengah secara horizontal (bukan seluruh lebar)
        left_x = x + int(0.15 * bw)
        right_x = x + int(0.85 * bw)

        # Clamp biar tidak keluar frame
        h, w, _ = frame_bgr.shape
        top_y = max(0, top_y)
        bottom_y = min(h, bottom_y)
        left_x = max(0, left_x)
        right_x = min(w, right_x)

        roi = frame_bgr[top_y:bottom_y, left_x:right_x]
        return roi, (left_x, top_y, right_x, bottom_y)



    # ========================
    # Bagian buffer & estimasi BPM
    # ========================

    def _update_signal_buffer(self, value: float, timestamp: float):
        """Menambahkan satu sample sinyal dan timestamp ke buffer sliding window."""
        self.signal_buffer.append(float(value))
        self.time_buffer.append(float(timestamp))

    def _estimate_fs_from_time_buffer(self):
        """
        Mengestimasi frekuensi sampling (fs) dari time_buffer
        menggunakan durasi real (bukan sekadar fps kamera).
        """
        if len(self.time_buffer) < 2:
            return None

        duration = self.time_buffer[-1] - self.time_buffer[0]
        if duration <= 0:
            return None

        fs = len(self.time_buffer) / duration
        return fs

    def _estimate_bpm_from_buffer(self):
        """
        Pipeline pemrosesan sinyal:
        - cek panjang waktu minimum
        - detrending
        - bandpass filter
        - FFT -> BPM
        - smoothing BPM
        """
        if len(self.signal_buffer) < 2:
            return None

        # Cek durasi minimal
        duration = self.time_buffer[-1] - self.time_buffer[0]
        if duration < self.min_seconds_for_bpm:
            return None

        sig = np.array(self.signal_buffer, dtype=np.float32)
        fs = self._estimate_fs_from_time_buffer()
        if fs is None or fs < 5.0:
            # Sampling terlalu rendah -> sinyal tidak valid
            return None

        # 1) Detrend dengan sliding window ~1 detik
        window_size = int(fs * 1.0)  # 1 detik
        if window_size < 3:
            window_size = 3
        sig_detrended = detrend_signal(sig, window_size=window_size)

        # 2) Bandpass filter 0.67 - 4.0 Hz
        sig_filtered = bandpass_filter(sig_detrended, fs=fs,
                                       low=0.67, high=4.0, order=4)

        # 3) FFT -> BPM
        bpm_raw, bpm_axis, spec_roi = get_bpm_from_fft(sig_filtered, fs=fs)

        if np.isnan(bpm_raw):
            return None

        self.current_bpm = bpm_raw

        # 4) Smoothing BPM (improvement untuk stabilitas nilai yang ditampilkan)
        if np.isnan(self.smooth_bpm):
            self.smooth_bpm = bpm_raw
        else:
            self.smooth_bpm = (
                self.alpha_bpm * self.smooth_bpm
                + (1.0 - self.alpha_bpm) * bpm_raw
            )

        return bpm_raw

    # ========================
    # Main loop real-time
    # ========================

    def run(self):
        """
        Menjalankan pipeline rPPG secara real-time.
        Tekan 'q' di jendela video untuk keluar.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Tidak dapat membuka kamera (webcam).")

        # Coba baca FPS dari kamera (kalau ada)
        fps_cam = cap.get(cv2.CAP_PROP_FPS)
        if fps_cam is not None and fps_cam > 1:
            self.fps_guess = fps_cam

            max_len = int(self.window_seconds * self.fps_guess)
            self.signal_buffer = deque(self.signal_buffer, maxlen=max_len)
            self.time_buffer = deque(self.time_buffer, maxlen=max_len)

        print("Tekan 'q' untuk keluar dari program.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari kamera.")
                break

            # Mirror agar lebih natural (seperti ngaca)
            frame = cv2.flip(frame, 1)

            # MediaPipe butuh RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Deteksi wajah
            face_bbox = self._detect_face_bbox(frame_rgb)

            timestamp = time.time()

            if face_bbox is not None:
                # Ambil ROI dahi
                roi, (x1, y1, x2, y2) = self._get_forehead_roi(frame, face_bbox)

                if roi.size != 0:
                    # Kanal Hijau = index 1 (BGR -> G)
                    green_channel = roi[:, :, 1]
                    mean_green = float(np.mean(green_channel))

                    self._update_signal_buffer(mean_green, timestamp)

                    # Gambar kotak ROI sebagai feedback visual
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Estimasi BPM dari buffer
            bpm_est = self._estimate_bpm_from_buffer()

            # Teks BPM
            if self.smooth_bpm is not None and not np.isnan(self.smooth_bpm):
                text_bpm = f"BPM: {self.smooth_bpm:.1f}"
            else:
                text_bpm = "BPM: --"

            cv2.putText(
                frame,
                text_bpm,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if self.show_debug:
                # Tambahan: tampilkan jumlah sample & durasi buffer
                if len(self.time_buffer) >= 2:
                    duration = self.time_buffer[-1] - self.time_buffer[0]
                    debug_text = f"Samples: {len(self.signal_buffer)} | Win: {duration:.1f}s"
                    cv2.putText(
                        frame,
                        debug_text,
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            cv2.imshow("rPPG – Real-time Heart Rate (press 'q' to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
