"""補正パラメータの定義。"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields

# --- 解析の基本設定 -------------------------------------------------------
F0_FLOOR = 65.0    # Hz. C2 くらい。44.1kHz なら CheapTrick の FFT が 2048 に収まる
F0_CEIL = 1000.0   # Hz. B5 より上。裏声も拾える
FRAME_PERIOD = 5.0  # ms. 解析・合成のフレーム間隔(= 200fps)


@dataclass
class CorrectionParams:
    """ピッチ補正の挙動を決めるパラメータ。

    strength / vibrato_keep / retune_ms の 3 つが「音の性格」を決める主役。
    残りは安全装置とアライメントの調整用。
    """

    # --- 補正の性格 ---
    strength: float = 0.85
    """ガイドへの追従度。0.0 = 無補正、1.0 = 完全にガイドの音程。
    0.7〜0.9 が自然。1.0 にすると機械的だが確実に合う。"""

    vibrato_keep: float = 1.0
    """自分の細かい揺れ(ビブラート・しゃくり)をどれだけ残すか。
    1.0 = 全部残す(自分の声らしさが保たれる)。
    0.0 = 揺れを消してまっすぐな音にする(不随意な震えを抑えたいとき)。"""

    drift_hz: float = 1.5
    """「ゆっくりしたズレ(ドリフト)」と「速い揺れ(ビブラート)」を分ける境目の周波数。
    これより遅い成分をガイドに合わせ、速い成分は vibrato_keep で扱う。
    ビブラートは通常 4〜7Hz なので 1.5Hz 前後が妥当。"""

    retune_ms: float = 80.0
    """補正量が変化するときの滑らかさ(ミリ秒)。
    小さいほど音の変わり目でカチッと決まり、大きいほど人間的にぬるっと動く。"""

    max_shift: float = 7.0
    """1 フレームで動かす上限(半音)。これを超えるズレは頭打ちにする。
    ピッチ推定ミス(オクターブ誤検出など)で音が破壊されるのを防ぐ安全装置。"""

    # --- 解析 ---
    f0_method: str = "dio"
    """F0 推定の方式。"dio"(高速・十分実用)/ "harvest"(高精度・数倍遅い)。"""

    f0_floor: float = F0_FLOOR
    f0_ceil: float = F0_CEIL
    frame_period: float = FRAME_PERIOD

    repair_octaves: bool = True
    """F0 推定のオクターブ誤検出を自動で直す。基本は ON のままでよい。"""

    # --- 時間合わせ ---
    offset_hint: float = 0.0
    """ガイドに対して録音がどれだけ遅れているかの初期見積り(秒)。
    ガイドの途中から録音した場合は -(開始位置) を入れる。実測で微調整される。"""

    offset_search: float = 5.0
    """offset_hint の周りを何秒ぶん探すか。"""

    use_dtw: bool = True
    """DTW で歌い出しや伸ばしのズレまで細かく合わせるか。
    OFF にすると全体の一律オフセットだけで合わせる(高速だがラフ)。"""

    dtw_margin: float = 1.5
    """DTW が許す時間のズレ幅(秒)。これ以上は動かさない。"""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "CorrectionParams":
        """未知のキーは黙って捨てる(UI からの余計なフィールドを許容するため)。"""
        if not data:
            return cls()
        known = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                continue
            if key in ("f0_method",):
                kwargs[key] = str(value)
            elif key in ("repair_octaves", "use_dtw"):
                kwargs[key] = bool(value)
            else:
                kwargs[key] = float(value)
        return cls(**kwargs)

    def validated(self) -> "CorrectionParams":
        """範囲外の値を安全側に丸めた新しいインスタンスを返す。"""

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        return CorrectionParams(
            strength=clamp(self.strength, 0.0, 1.0),
            vibrato_keep=clamp(self.vibrato_keep, 0.0, 1.5),
            drift_hz=clamp(self.drift_hz, 0.2, 8.0),
            retune_ms=clamp(self.retune_ms, 5.0, 1000.0),
            max_shift=clamp(self.max_shift, 0.5, 24.0),
            f0_method=self.f0_method if self.f0_method in ("dio", "harvest") else "dio",
            f0_floor=clamp(self.f0_floor, 40.0, 200.0),
            f0_ceil=clamp(self.f0_ceil, 300.0, 2000.0),
            frame_period=clamp(self.frame_period, 1.0, 10.0),
            repair_octaves=self.repair_octaves,
            offset_hint=clamp(self.offset_hint, -600.0, 600.0),
            offset_search=clamp(self.offset_search, 0.1, 30.0),
            use_dtw=self.use_dtw,
            dtw_margin=clamp(self.dtw_margin, 0.05, 10.0),
        )
