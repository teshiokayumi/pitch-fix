"""F0(基本周波数)の推定と、半音スケールでの後処理。"""

from __future__ import annotations

import numpy as np
import pyworld as pw
from scipy.ndimage import median_filter

from .params import F0_CEIL, F0_FLOOR, FRAME_PERIOD

A4_HZ = 440.0
A4_MIDI = 69.0


def extract_f0(
    x: np.ndarray,
    fs: int,
    method: str = "dio",
    frame_period: float = FRAME_PERIOD,
    f0_floor: float = F0_FLOOR,
    f0_ceil: float = F0_CEIL,
) -> tuple[np.ndarray, np.ndarray]:
    """波形から F0 列と時刻列を返す。無声区間の F0 は 0。

    dio は速く、harvest は遅いが正確。どちらも stonemask で精密化する。
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    if method == "harvest":
        f0, t = pw.harvest(x, fs, f0_floor=f0_floor, f0_ceil=f0_ceil,
                           frame_period=frame_period)
    else:
        f0, t = pw.dio(x, fs, f0_floor=f0_floor, f0_ceil=f0_ceil,
                       frame_period=frame_period)
    f0 = pw.stonemask(x, f0, t, fs)
    return np.ascontiguousarray(f0), np.ascontiguousarray(t)


def hz_to_midi(f0: np.ndarray) -> np.ndarray:
    """Hz を MIDI ノート番号(半音単位の実数)へ。無声(0Hz)は NaN。"""
    midi = np.full(f0.shape, np.nan, dtype=np.float64)
    voiced = f0 > 0
    midi[voiced] = A4_MIDI + 12.0 * np.log2(f0[voiced] / A4_HZ)
    return midi


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """MIDI ノート番号を Hz へ。NaN は 0Hz(無声)。"""
    f0 = np.zeros(midi.shape, dtype=np.float64)
    voiced = np.isfinite(midi)
    f0[voiced] = A4_HZ * np.exp2((midi[voiced] - A4_MIDI) / 12.0)
    return f0


def voiced_segments(midi: np.ndarray) -> list[tuple[int, int]]:
    """有声が連続している区間 [start, end) の一覧を返す。"""
    voiced = np.isfinite(midi)
    if not voiced.any():
        return []
    # 端に False を足して立ち上がり/立ち下がりを取る
    padded = np.concatenate(([False], voiced, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def repair_octaves(midi: np.ndarray, fps: float, win_s: float = 0.25) -> np.ndarray:
    """オクターブ誤検出を直す。

    F0 推定は倍音や低域に引っ張られて、ときどき 1 オクターブ跳んだ値を出す。
    局所メディアンを基準に、±12/±24 半音ずらして一番近い候補に寄せる。
    実際の跳躍(オクターブ上のフレーズ)は局所メディアン自体が追従するので壊さない。
    """
    out = midi.copy()
    win = max(3, int(round(win_s * fps)))
    if win % 2 == 0:
        win += 1

    for start, end in voiced_segments(out):
        seg = out[start:end]
        if seg.size < 5:
            continue
        size = min(win, seg.size if seg.size % 2 == 1 else seg.size - 1)
        if size < 3:
            continue
        for _ in range(2):  # 1 回目で基準が整い、2 回目で残りを拾う
            ref = median_filter(seg, size=size, mode="nearest")
            shift = np.clip(np.round((ref - seg) / 12.0), -2, 2)
            if not shift.any():
                break
            seg = seg + 12.0 * shift
        out[start:end] = seg
    return out


def median_note(midi: np.ndarray) -> float:
    """有声部分の中央値(MIDI)。音域の目安表示に使う。"""
    voiced = midi[np.isfinite(midi)]
    return float(np.median(voiced)) if voiced.size else float("nan")
