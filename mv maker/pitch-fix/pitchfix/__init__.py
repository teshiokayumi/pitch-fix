"""pitchfix — ガイドボーカルに合わせて歌声のピッチを補正するツール。

自分の歌の「音程」だけをガイドに寄せ、リズム・声質・ビブラートはそのまま残す。
WORLD ボコーダ(pyworld)を使うのでフォルマントが保たれ、ケロケロ声にならない。
"""

from .params import CorrectionParams  # noqa: F401
from .pipeline import correct_take  # noqa: F401

__all__ = ["CorrectionParams", "correct_take"]
