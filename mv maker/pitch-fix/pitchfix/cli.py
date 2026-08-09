"""コマンドラインから使う入口。

    python -m pitchfix.cli guide.wav take.wav -o fixed.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .params import CorrectionParams
from .pipeline import correct_take


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pitchfix",
        description="ガイドボーカルに合わせて歌声のピッチを補正します。",
    )
    p.add_argument("guide", help="ガイドボーカルの音声ファイル(ボーカルのみ)")
    p.add_argument("take", help="補正したい自分の歌の音声ファイル")
    p.add_argument("-o", "--output", default=None, help="出力先 WAV(既定: <take>_fixed.wav)")

    p.add_argument("--strength", type=float, default=0.85,
                   help="ガイドへの追従度 0.0-1.0(既定 0.85)")
    p.add_argument("--vibrato-keep", type=float, default=1.0,
                   help="自分の細かい揺れを残す度合い 0.0-1.0(既定 1.0)")
    p.add_argument("--retune-ms", type=float, default=80.0,
                   help="補正の追従の速さ(ミリ秒、既定 80)")
    p.add_argument("--drift-hz", type=float, default=1.5,
                   help="ドリフトとビブラートを分ける周波数(既定 1.5)")
    p.add_argument("--max-shift", type=float, default=7.0,
                   help="動かす上限(半音、既定 7)")

    p.add_argument("--f0-method", choices=("dio", "harvest"), default="dio",
                   help="ピッチ推定の方式(既定 dio)")
    p.add_argument("--offset", type=float, default=0.0,
                   help="録音の遅れの初期値(秒)。ガイド途中から歌った場合は -開始位置")
    p.add_argument("--no-dtw", action="store_true",
                   help="DTW による細かい時間合わせを使わない")
    p.add_argument("--json", action="store_true", help="結果のサマリーを JSON で出力")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    take_path = Path(args.take)
    output = Path(args.output) if args.output else take_path.with_name(
        take_path.stem + "_fixed.wav"
    )

    params = CorrectionParams(
        strength=args.strength,
        vibrato_keep=args.vibrato_keep,
        retune_ms=args.retune_ms,
        drift_hz=args.drift_hz,
        max_shift=args.max_shift,
        f0_method=args.f0_method,
        offset_hint=args.offset,
        use_dtw=not args.no_dtw,
    )

    def progress(fraction: float, message: str) -> None:
        if not args.json:
            print(f"\r[{int(fraction * 100):3d}%] {message:<24}", end="", file=sys.stderr)

    try:
        result = correct_take(args.guide, take_path, params, progress=progress)
    except Exception as exc:  # ユーザー向けに 1 行で伝える
        print(f"\nエラー: {exc}", file=sys.stderr)
        return 1

    if not args.json:
        print(file=sys.stderr)
    result.save(output)

    summary = result.summary()
    if args.json:
        print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    else:
        print(f"書き出し: {output}")
        print(f"  ガイドとのずれ(補正前) 中央値 {summary['median_error']:.2f} 半音 / "
              f"90% {summary['p90_error']:.2f} 半音")
        print(f"  補正後の残差 中央値 {summary['residual_median']:.2f} 半音")
        print(f"  時間オフセット {summary['offset']:+.3f} 秒 "
              f"(信頼度 {summary['offset_confidence']:.1f})")
        print(f"  ガイド参照できた割合 {summary['coverage'] * 100:.1f}%")
        if summary["clamp_ratio"] > 0.05:
            print(f"  ※ {summary['clamp_ratio'] * 100:.0f}% のフレームが上限"
                  f"({args.max_shift} 半音)で頭打ちになりました。"
                  "オクターブ違いで歌っていないか確認してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
