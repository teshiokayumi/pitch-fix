"""ガイドと録音の時間合わせ。

やることは 2 段階:
  1. 全体オフセット推定 — 録音がガイドから何秒ずれて始まったか(録音レイテンシ、
     カウントインのずれ、ガイドの途中から歌い始めた場合など)を相互相関で求める。
  2. DTW による細かい歪み補正 — 歌い出しの突っ込み/もたり、伸ばしの長さの違いを吸収する。

重要なのは「録音の時間軸は一切いじらない」こと。求めるのは
「録音の時刻 t のとき、ガイドのどの時刻を目標ピッチとして読むべきか」という対応表だけ。
本人のリズムやタイム感はそのまま残る。
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import correlate

FEATURE_SR = 22050
FEATURE_HOP_S = 0.01  # 100 fps


def _resample(x: np.ndarray, fs: int, target_sr: int = FEATURE_SR) -> np.ndarray:
    if fs == target_sr:
        return np.ascontiguousarray(x, dtype=np.float32)
    return librosa.resample(
        np.ascontiguousarray(x, dtype=np.float32), orig_sr=fs, target_sr=target_sr
    )


def _hop() -> int:
    return max(1, int(round(FEATURE_SR * FEATURE_HOP_S)))


def feature_fps() -> float:
    return FEATURE_SR / _hop()


def timbre_features(x: np.ndarray, fs: int) -> np.ndarray:
    """DTW 用の音色特徴 (n_features, n_frames)。

    MFCC の 0 次(全体の音量)を捨てているので、ガイドと録音で音量やマイクが
    違っても対応が取れる。同じ歌詞を歌っていれば母音の並びで揃う。
    """
    y = _resample(x, fs)
    mfcc = librosa.feature.mfcc(y=y, sr=FEATURE_SR, n_mfcc=20,
                               n_fft=1024, hop_length=_hop())
    mfcc = mfcc[1:]
    mfcc -= mfcc.mean(axis=1, keepdims=True)
    mfcc /= mfcc.std(axis=1, keepdims=True) + 1e-8
    return np.ascontiguousarray(mfcc, dtype=np.float32)


def onset_envelope(x: np.ndarray, fs: int) -> np.ndarray:
    """発音の立ち上がりの強さ。オフセット推定用。"""
    y = _resample(x, fs)
    env = librosa.onset.onset_strength(y=y, sr=FEATURE_SR, hop_length=_hop())
    env = env - env.mean()
    std = env.std()
    return env / std if std > 1e-8 else env


def melodic_contour(
    midi: np.ndarray,
    frame_period: float,
    n_frames: int,
) -> np.ndarray:
    """旋律の上下だけを取り出した信号(n_frames サンプル、100fps)。

    局所メディアンを引くことで「キーが何か」「その人の音域」「全体のドリフト」を
    落とし、音の上がり下がりの形だけを残す。音程が多少ずれていても形は残るので、
    オフセット推定の手がかりとして使える。
    """
    fps = feature_fps()
    step = frame_period / 1000.0
    idx = np.clip(np.round(np.arange(n_frames) / fps / step).astype(np.int64),
                  0, max(0, midi.size - 1))
    m = midi[idx] if midi.size else np.full(n_frames, np.nan)

    voiced = np.isfinite(m)
    if voiced.sum() < 10:
        return np.zeros(n_frames)

    # メディアンフィルタのために無声を線形補間で埋める(出力では捨てる)
    positions = np.arange(n_frames)
    filled = np.interp(positions, positions[voiced], m[voiced])
    win = int(4.0 * fps) | 1
    base = median_filter(filled, size=min(win, n_frames if n_frames % 2 else n_frames - 1),
                         mode="nearest") if n_frames >= 3 else filled

    contour = np.clip(filled - base, -12.0, 12.0)
    contour[~voiced] = 0.0
    std = contour.std()
    return contour / std if std > 1e-8 else contour


def _zscore_in_window(values: np.ndarray, window: np.ndarray) -> np.ndarray:
    """探索範囲内の分布で標準化する。異種のスコアを足し合わせるため。"""
    inside = values[window]
    spread = inside.std()
    if spread <= 1e-8:
        return np.zeros_like(values)
    return (values - inside.mean()) / spread


def estimate_offset(
    guide: np.ndarray,
    take: np.ndarray,
    fs_guide: int,
    fs_take: int,
    hint: float = 0.0,
    search: float = 5.0,
    guide_midi: np.ndarray | None = None,
    take_midi: np.ndarray | None = None,
    frame_period: float = 5.0,
) -> tuple[float, float]:
    """録音がガイドからどれだけ遅れているか(秒)と、その確からしさを返す。

    戻り値の offset は「guide_time = take_time - offset」という向きで定義する。

    発音の立ち上がり(リズム)だけで合わせると、規則的な旋律では 1 小節ずれた
    ところにも同じくらい高いピークが立って取り違える。そこで旋律の上下の形も
    同時に相関させ、両方の合意が取れた位置を選ぶ。
    """
    env_g = onset_envelope(guide, fs_guide)
    env_t = onset_envelope(take, fs_take)
    if env_g.size < 4 or env_t.size < 4:
        return hint, 0.0

    fps = feature_fps()
    lags = np.arange(-(env_g.size - 1), env_t.size, dtype=np.float64)
    # correlate の定義より corr[k] = Σ env_t[l] * env_g[l - lag] なので、
    # ピークの lag がそのまま「録音が何フレーム遅れているか」になる。
    lag_s = lags / fps

    window = np.abs(lag_s - hint) <= search
    if not window.any():
        return hint, 0.0

    onset_corr = correlate(env_t, env_g, mode="full", method="fft")
    score = _zscore_in_window(onset_corr, window)

    if guide_midi is not None and take_midi is not None:
        con_g = melodic_contour(guide_midi, frame_period, env_g.size)
        con_t = melodic_contour(take_midi, frame_period, env_t.size)
        if con_g.any() and con_t.any():
            contour_corr = correlate(con_t, con_g, mode="full", method="fft")
            # 旋律の形のほうが識別力が高いので重く見る
            score = score + 2.0 * _zscore_in_window(contour_corr, window)

    best = int(np.argmax(np.where(window, score, -np.inf)))
    inside = score[window]
    spread = inside.std()
    confidence = float((score[best] - inside.mean()) / spread) if spread > 1e-8 else 0.0
    return float(lag_s[best]), confidence


def _dtw_chunk(feat_g: np.ndarray, feat_t: np.ndarray) -> np.ndarray:
    """1 チャンクぶんの DTW。録音フレームごとの「対応するガイドフレーム」を返す。"""
    _, wp = librosa.sequence.dtw(X=feat_g, Y=feat_t, backtrack=True,
                                 global_constraints=False)
    n_take = feat_t.shape[1]
    sums = np.zeros(n_take)
    counts = np.zeros(n_take)
    # 1 つの録音フレームに複数のガイドフレームが対応しうるので平均を取る
    np.add.at(sums, wp[:, 1], wp[:, 0].astype(np.float64))
    np.add.at(counts, wp[:, 1], 1.0)
    valid = counts > 0
    out = np.full(n_take, np.nan)
    out[valid] = sums[valid] / counts[valid]
    return out


def build_warp(
    guide: np.ndarray,
    take: np.ndarray,
    fs_guide: int,
    fs_take: int,
    offset: float,
    use_dtw: bool = True,
    margin: float = 1.5,
    chunk_s: float = 20.0,
    hop_s: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """録音時刻 → ガイド時刻 の対応表を返す。

    Returns
    -------
    (take_times, guide_times) : いずれも 100fps 相当の等間隔配列。単調非減少。
    """
    fps = feature_fps()
    feat_t = timbre_features(take, fs_take)
    n_take = feat_t.shape[1]
    take_times = np.arange(n_take) / fps
    linear = take_times - offset

    if not use_dtw:
        return take_times, linear

    feat_g = timbre_features(guide, fs_guide)
    n_guide = feat_g.shape[1]
    if n_guide < 20 or n_take < 20:
        return take_times, linear

    margin_f = max(4, int(round(margin * fps)))
    chunk_f = max(50, int(round(chunk_s * fps)))
    hop_f = max(10, int(round(hop_s * fps)))

    acc = np.zeros(n_take)
    weight = np.zeros(n_take)

    # チャンクを進めながらオフセットを更新していく。曲の途中でテンポが少しずつ
    # ずれても追随できる(全体を 1 つのオフセットで見ると後半で外れる)。
    running_off = offset * fps

    for i0 in range(0, n_take, hop_f):
        i1 = min(n_take, i0 + chunk_f)
        if i1 - i0 < 20:
            break
        g0 = int(np.clip(round(i0 - running_off) - margin_f, 0, n_guide))
        g1 = int(np.clip(round(i1 - running_off) + margin_f, 0, n_guide))
        if g1 - g0 < 20:
            continue

        local = _dtw_chunk(feat_g[:, g0:g1], feat_t[:, i0:i1]) + g0
        valid = np.isfinite(local)
        if valid.sum() < 10:
            continue

        # チャンク中央を厚く、端を薄く重み付けして重なりを滑らかに繋ぐ
        n = i1 - i0
        w = np.maximum(1.0 - np.abs(np.linspace(-1.0, 1.0, n)), 0.05)
        w = np.where(valid, w, 0.0)

        acc[i0:i1] += np.nan_to_num(local, nan=0.0) * w
        weight[i0:i1] += w

        # 次のチャンクの探索窓は、このチャンクの後半で実測したずれを基準にする
        latter = np.arange(i0, i1) >= i0 + n // 2
        sample = latter & valid
        if sample.sum() >= 10:
            measured = float(np.median(np.arange(i0, i1)[sample] - local[sample]))
            if abs(measured - running_off) <= margin_f:
                running_off = measured

        if i1 >= n_take:
            break

    mapped = np.full(n_take, np.nan)
    have = weight > 0
    mapped[have] = acc[have] / weight[have]

    # DTW が取れなかったフレームは一律オフセットで埋める
    mapped = np.where(np.isfinite(mapped), mapped, linear * fps)

    mapped = gaussian_filter1d(mapped, sigma=max(1.0, 0.05 * fps), mode="nearest")
    mapped = np.maximum.accumulate(mapped)  # 時間は巻き戻らない

    return take_times, mapped / fps
