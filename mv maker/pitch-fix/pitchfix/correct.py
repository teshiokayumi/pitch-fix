"""目標ピッチ曲線の生成と、補正後 F0 の計算。

このモジュールが「どう直すか」の心臓部。考え方はこう:

    自分の歌の音程 = ゆっくりした成分(u_slow) + 速い揺れ(u_fine)

ゆっくりした成分は「その音をどこで取っているか」= 直したいズレ。
速い揺れは「ビブラート・しゃくり・語尾の表情」= 残したい個性。

この 2 つを分けてから、ゆっくりした成分だけをガイドに寄せる。だから
「音程は合っているのに歌が平坦になった」ということが起きにくい。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .analysis import voiced_segments
from .params import CorrectionParams

# ガウシアン平滑のカットオフ周波数 fc [Hz] に対応する σ [秒]。
# ガウス窓の周波数応答が -3dB になる点から。
_SIGMA_PER_HZ = 0.1874


@dataclass
class CorrectionStats:
    """補正の中身を人間が確認するための集計値。"""

    voiced_frames: int = 0
    covered_frames: int = 0        # ガイドの参照が取れたフレーム数
    median_error: float = 0.0      # 補正前のズレの中央値(半音、絶対値)
    p90_error: float = 0.0         # 同 90 パーセンタイル
    max_error: float = 0.0
    clamped_frames: int = 0        # max_shift で頭打ちになったフレーム数
    residual_median: float = 0.0   # 補正後に残ったズレの中央値(半音)

    @property
    def coverage(self) -> float:
        return self.covered_frames / self.voiced_frames if self.voiced_frames else 0.0

    @property
    def clamp_ratio(self) -> float:
        return self.clamped_frames / self.covered_frames if self.covered_frames else 0.0

    def to_dict(self) -> dict:
        return {
            "voiced_frames": self.voiced_frames,
            "covered_frames": self.covered_frames,
            "coverage": round(self.coverage, 4),
            "median_error": round(self.median_error, 3),
            "p90_error": round(self.p90_error, 3),
            "max_error": round(self.max_error, 3),
            "clamped_frames": self.clamped_frames,
            "clamp_ratio": round(self.clamp_ratio, 4),
            "residual_median": round(self.residual_median, 3),
        }


def resample_guide_midi(
    guide_midi: np.ndarray,
    guide_frame_period: float,
    take_times: np.ndarray,
    warp_take_times: np.ndarray,
    warp_guide_times: np.ndarray,
) -> np.ndarray:
    """録音のフレーム時刻ごとに、対応するガイドの音程を引いてくる。

    線形補間ではなく最近傍で引く。有声と無声の境目をまたいで補間すると、
    実在しない中間の音程が生まれて音の変わり目が濁るため。
    """
    guide_times_for_take = np.interp(take_times, warp_take_times, warp_guide_times)
    step = guide_frame_period / 1000.0
    idx = np.round(guide_times_for_take / step).astype(np.int64)

    out = np.full(take_times.shape, np.nan)
    inside = (idx >= 0) & (idx < guide_midi.size)
    out[inside] = guide_midi[idx[inside]]
    return out


def _fill_nearest(values: np.ndarray) -> np.ndarray:
    """NaN を最も近い有効値で埋める(端は端の値で延長)。全部 NaN なら None。"""
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return None
    positions = np.arange(values.size)
    nearest = np.searchsorted(valid, positions)
    nearest = np.clip(nearest, 0, valid.size - 1)
    prev = np.clip(nearest - 1, 0, valid.size - 1)
    take_prev = np.abs(positions - valid[prev]) <= np.abs(positions - valid[nearest])
    chosen = np.where(take_prev, valid[prev], valid[nearest])
    return values[chosen]


def build_corrected_midi(
    user_midi: np.ndarray,
    target_midi: np.ndarray,
    fps: float,
    params: CorrectionParams,
) -> tuple[np.ndarray, CorrectionStats]:
    """補正後の音程曲線(MIDI)と統計を返す。

    user_midi の無声フレームは触らない(無い音は作れない)。
    target_midi が NaN のフレームでは補正量をなめらかに 0 へ戻す。
    """
    corrected = user_midi.copy()
    stats = CorrectionStats()

    sigma_slow = max(1.0, _SIGMA_PER_HZ / params.drift_hz * fps)
    sigma_retune = max(1.0, params.retune_ms / 1000.0 * fps)

    errors: list[np.ndarray] = []
    residuals: list[np.ndarray] = []

    for start, end in voiced_segments(user_midi):
        u = user_midi[start:end]
        g = target_midi[start:end]
        stats.voiced_frames += u.size

        if u.size < 3:
            continue

        available = np.isfinite(g)
        if not available.any():
            continue
        stats.covered_frames += int(available.sum())

        g_filled = _fill_nearest(g)
        if g_filled is None:
            continue

        # ゆっくりした成分と速い揺れに分解する
        u_slow = gaussian_filter1d(u, sigma=sigma_slow, mode="nearest")
        u_fine = u - u_slow
        g_slow = gaussian_filter1d(g_filled, sigma=sigma_slow, mode="nearest")

        delta = g_slow - u_slow

        clamped = np.abs(delta) > params.max_shift
        stats.clamped_frames += int((clamped & available).sum())
        delta = np.clip(delta, -params.max_shift, params.max_shift)

        # ガイドが無い区間では補正量をじわっと 0 に戻す(段差を作らない)
        coverage_w = gaussian_filter1d(available.astype(np.float64),
                                       sigma=sigma_retune, mode="nearest")
        delta = delta * coverage_w

        delta = gaussian_filter1d(delta, sigma=sigma_retune, mode="nearest")
        delta *= params.strength

        corrected[start:end] = u_slow + params.vibrato_keep * u_fine + delta

        if available.any():
            errors.append(np.abs(g[available] - u[available]))
            residuals.append(np.abs(g[available] - corrected[start:end][available]))

    if errors:
        all_errors = np.concatenate(errors)
        stats.median_error = float(np.median(all_errors))
        stats.p90_error = float(np.percentile(all_errors, 90))
        stats.max_error = float(all_errors.max())
    if residuals:
        stats.residual_median = float(np.median(np.concatenate(residuals)))

    return corrected, stats
