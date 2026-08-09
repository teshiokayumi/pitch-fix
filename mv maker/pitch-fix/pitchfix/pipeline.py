"""ガイド + 録音 → 補正済み音声、までの一本道。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import align, analysis, synth
from .audio import load_mono, normalize_if_clipping, save_wav
from .correct import CorrectionStats, build_corrected_midi, resample_guide_midi
from .params import CorrectionParams

ProgressFn = Callable[[float, str], None]


@dataclass
class CorrectionResult:
    audio: np.ndarray
    fs: int
    stats: CorrectionStats
    offset: float
    offset_confidence: float
    gain: float
    curves: dict = field(default_factory=dict)

    def save(self, path: str | Path, subtype: str = "PCM_24") -> None:
        save_wav(path, self.audio, self.fs, subtype=subtype)

    def summary(self) -> dict:
        return {
            "offset": round(self.offset, 3),
            "offset_confidence": round(self.offset_confidence, 2),
            "output_gain": round(self.gain, 3),
            "duration": round(self.audio.size / self.fs, 2),
            **self.stats.to_dict(),
        }


def _decimate_curve(values: np.ndarray, max_points: int) -> list[float | None]:
    """グラフ描画用に間引く。NaN は None にして JSON で「無声」を表す。"""
    if values.size > max_points:
        idx = np.linspace(0, values.size - 1, max_points).round().astype(np.int64)
        values = values[idx]
    return [None if not np.isfinite(v) else round(float(v), 3) for v in values]


def correct_take(
    guide: str | Path | tuple[np.ndarray, int],
    take: str | Path | tuple[np.ndarray, int],
    params: CorrectionParams | None = None,
    progress: ProgressFn | None = None,
    curve_points: int = 1500,
) -> CorrectionResult:
    """ガイドに合わせて録音のピッチを補正する。

    guide / take はファイルパス、または (波形, サンプリング周波数) のタプル。
    """
    params = (params or CorrectionParams()).validated()

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, fraction)), message)

    report(0.02, "音声を読み込み中")
    guide_x, guide_fs = guide if isinstance(guide, tuple) else load_mono(guide)
    take_x, take_fs = take if isinstance(take, tuple) else load_mono(take)

    if take_x.size < take_fs * 0.5:
        raise ValueError("録音が短すぎます(0.5 秒未満)。")
    if guide_x.size < guide_fs * 0.5:
        raise ValueError("ガイド音源が短すぎます(0.5 秒未満)。")

    fp = params.frame_period
    fps = 1000.0 / fp

    report(0.08, "ガイドのピッチを解析中")
    guide_f0, _ = analysis.extract_f0(
        guide_x, guide_fs, method=params.f0_method, frame_period=fp,
        f0_floor=params.f0_floor, f0_ceil=params.f0_ceil,
    )
    guide_midi = analysis.hz_to_midi(guide_f0)
    if params.repair_octaves:
        guide_midi = analysis.repair_octaves(guide_midi, fps)

    if not np.isfinite(guide_midi).any():
        raise ValueError(
            "ガイド音源から音程を検出できませんでした。"
            "伴奏入りのミックスではなく、ボーカルのみの音源を使ってください。"
        )

    report(0.25, "録音のピッチを解析中")
    take_f0, take_t = analysis.extract_f0(
        take_x, take_fs, method=params.f0_method, frame_period=fp,
        f0_floor=params.f0_floor, f0_ceil=params.f0_ceil,
    )
    take_midi_raw = analysis.hz_to_midi(take_f0)
    take_midi = analysis.repair_octaves(take_midi_raw, fps) if params.repair_octaves else take_midi_raw

    if not np.isfinite(take_midi).any():
        raise ValueError(
            "録音から音程を検出できませんでした。"
            "マイクが拾えているか、音量が小さすぎないか確認してください。"
        )

    report(0.42, "ガイドと録音の時間を合わせ中")
    offset, confidence = align.estimate_offset(
        guide_x, take_x, guide_fs, take_fs,
        hint=params.offset_hint, search=params.offset_search,
        guide_midi=guide_midi, take_midi=take_midi, frame_period=fp,
    )
    # 相関のピークが立っていないときは推定を信用せず、指示された値を使う
    if confidence < 2.0:
        offset = params.offset_hint

    warp_take_t, warp_guide_t = align.build_warp(
        guide_x, take_x, guide_fs, take_fs, offset,
        use_dtw=params.use_dtw, margin=params.dtw_margin,
    )

    report(0.55, "目標のピッチ曲線を作成中")
    target_midi = resample_guide_midi(guide_midi, fp, take_t, warp_take_t, warp_guide_t)
    corrected_midi, stats = build_corrected_midi(take_midi, target_midi, fps, params)

    report(0.62, "再合成の準備中")
    take_f0_fixed = analysis.midi_to_hz(take_midi)   # オクターブ修正を反映した解析用 F0
    corrected_f0 = analysis.midi_to_hz(corrected_midi)

    def synth_progress(fraction: float, message: str) -> None:
        report(0.65 + 0.32 * fraction, message)

    audio = synth.resynthesize(
        take_x, take_fs, take_f0_fixed, take_t, corrected_f0,
        frame_period=fp, f0_floor=params.f0_floor, progress=synth_progress,
    )
    audio, gain = normalize_if_clipping(audio)

    report(0.99, "仕上げ中")
    curves = {
        "frame_period": fp,
        "duration": take_x.size / take_fs,
        "points": min(curve_points, take_midi.size),
        "original": _decimate_curve(take_midi, curve_points),
        "target": _decimate_curve(target_midi, curve_points),
        "corrected": _decimate_curve(corrected_midi, curve_points),
    }

    report(1.0, "完了")
    return CorrectionResult(
        audio=audio, fs=take_fs, stats=stats, offset=offset,
        offset_confidence=confidence, gain=gain, curves=curves,
    )
