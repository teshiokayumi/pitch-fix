"""音声ファイルの読み書き。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """モノラル float64 として読み込む。ステレオは平均でダウンミックス。"""
    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    return np.ascontiguousarray(mono), int(fs)


def save_wav(path: str | Path, x: np.ndarray, fs: int, subtype: str = "PCM_24") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float64), fs, subtype=subtype)


def normalize_if_clipping(x: np.ndarray, ceiling: float = 0.99) -> tuple[np.ndarray, float]:
    """クリップする場合だけ音量を下げる。それ以外は元のレベルを保つ。

    戻り値は (波形, かけたゲイン)。ゲインが 1.0 未満なら下げたということ。
    """
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= ceiling or peak == 0.0:
        return x, 1.0
    gain = ceiling / peak
    return x * gain, gain
