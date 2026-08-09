"""WORLD ボコーダによる再合成。

ポイントは、スペクトル包絡(= フォルマント、声の音色)を **元の F0 で** 解析し、
合成のときだけ F0 を差し替えること。包絡はそのまま使われるので、音程を上下しても
声が高いおもちゃ声になったり野太くなったりしない。ここが単純なリサンプリングとの差。

長い曲でもメモリが破裂しないよう、時間方向にチャンク分割して処理する。
解析は毎回フル波形を渡し、フレーム時刻だけを切り出すので、チャンク境界でも
窓が欠けることがない。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pyworld as pw

ProgressFn = Callable[[float, str], None]


def _find_cut_frames(f0: np.ndarray, chunk_frames: int, search_frames: int) -> list[int]:
    """チャンクの境界を、なるべく無声(息継ぎ・子音・無音)の位置に置く。

    WORLD は呼び出しごとに位相を作り直すので、有声の途中で切って繋ぐと
    重ね合わせでコムフィルタ的な濁りが出る。無声で切ればそれが起きない。
    """
    total = f0.size
    cuts = [0]
    while cuts[-1] + chunk_frames < total:
        target = cuts[-1] + chunk_frames
        lo = max(cuts[-1] + chunk_frames // 2, target - search_frames)
        hi = min(total - 1, target + search_frames)
        cut = target
        if hi > lo:
            unvoiced = np.flatnonzero(f0[lo:hi] <= 0)
            if unvoiced.size:
                cut = int(lo + unvoiced[np.argmin(np.abs(unvoiced + lo - target))])
        if cut <= cuts[-1]:
            cut = target
        cuts.append(int(cut))
    cuts.append(total)
    return cuts


def resynthesize(
    x: np.ndarray,
    fs: int,
    f0_analysis: np.ndarray,
    t: np.ndarray,
    f0_new: np.ndarray,
    frame_period: float,
    f0_floor: float,
    chunk_s: float = 30.0,
    crossfade_ms: float = 30.0,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """f0_new の音程で x を再合成する。長さは x と同じ。"""
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0_analysis = np.ascontiguousarray(f0_analysis, dtype=np.float64)
    f0_new = np.ascontiguousarray(f0_new, dtype=np.float64)
    t = np.ascontiguousarray(t, dtype=np.float64)

    # 元が無声のところは無声のまま。有声を勝手に生やさない。
    f0_new = np.where(f0_analysis > 0, f0_new, 0.0)
    f0_new = np.where(np.isfinite(f0_new), f0_new, 0.0)
    f0_new = np.clip(f0_new, 0.0, fs / 2.0 - 1.0)
    # 極端に低い値は WORLD が扱えないので無声扱いにする
    f0_new = np.where((f0_new > 0) & (f0_new < f0_floor * 0.5), 0.0, f0_new)

    fft_size = pw.get_cheaptrick_fft_size(fs, f0_floor)
    total_frames = f0_analysis.size
    chunk_frames = max(200, int(round(chunk_s * 1000.0 / frame_period)))
    search_frames = max(20, int(round(3000.0 / frame_period)))
    fade_frames = max(1, int(round(crossfade_ms / frame_period)))

    cuts = _find_cut_frames(f0_analysis, chunk_frames, search_frames)
    n_chunks = len(cuts) - 1

    out = np.zeros(x.size + fs, dtype=np.float64)
    weight = np.zeros_like(out)
    samples_per_frame = fs * frame_period / 1000.0

    for i in range(n_chunks):
        a = max(0, cuts[i] - fade_frames)
        b = min(total_frames, cuts[i + 1] + fade_frames)
        if b - a < 3:
            continue

        f0_chunk = np.ascontiguousarray(f0_analysis[a:b])
        t_chunk = np.ascontiguousarray(t[a:b])

        sp = pw.cheaptrick(x, f0_chunk, t_chunk, fs,
                           f0_floor=f0_floor, fft_size=fft_size)
        ap = pw.d4c(x, f0_chunk, t_chunk, fs, fft_size=fft_size)
        y = pw.synthesize(np.ascontiguousarray(f0_new[a:b]), sp, ap, fs, frame_period)
        del sp, ap

        # 前後の重なりだけ直線でクロスフェード
        w = np.ones(y.size)
        ramp = min(int(round(fade_frames * samples_per_frame)), y.size // 2)
        if ramp > 1:
            if a > 0:
                w[:ramp] = np.linspace(0.0, 1.0, ramp)
            if b < total_frames:
                w[-ramp:] = np.linspace(1.0, 0.0, ramp)

        start = int(round(a * samples_per_frame))
        stop = min(out.size, start + y.size)
        n = stop - start
        if n <= 0:
            continue
        out[start:stop] += y[:n] * w[:n]
        weight[start:stop] += w[:n]

        if progress:
            progress((i + 1) / n_chunks, f"再合成中 {i + 1}/{n_chunks}")

    active = weight > 1e-6
    out[active] /= weight[active]
    return out[:x.size]
