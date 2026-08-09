# ブラウザだけで「音に反応するかわいいMV」をフル尺生成する ─ Canvas + Web Audio + MediaRecorder

## はじめに

知り合いが「背景イラストのスライドショーの上で、ドット絵のイコライザーが音楽に合わせて跳ねる」という30秒のかわいいMVを作っているのを見て、こう思いました。

**「これ、ブラウザのAPIだけで曲のフル尺を自分で作れるのでは?」**

結論から言うと、**サーバーもビルドツールも動画編集ソフトも一切なし**、`index.html` 1ファイルだけで作れました。この記事ではその仕組みと、実装でハマったポイントを紹介します。

### 作ったもの

- 16:9の画像を複数枚 + MP3を1曲アップロード
- 背景に画像がフェード/スライドで切り替わるスライドショー
- 画面下部で♡や🌸や🐾のドット絵が音に反応して積み上がるイコライザー
- **曲の最初から最後まで(フル尺)を1080pのMP4/WebMとして書き出し**

技術スタックは Vanilla JS + Canvas 2D + Web Audio API + MediaRecorder のみ。完全クライアントサイドなので、音源が外部に送信されることもありません。

## 全体アーキテクチャ

データの流れはこうなっています。

```
MP3 ─ decodeAudioData → AudioBuffer
                           │
                 AudioBufferSourceNode
                           │
                     AnalyserNode ──── getByteFrequencyData() ──→ Canvas描画
                           │                                        │
                       GainNode ──┬── AudioContext.destination     │
                                  │      (スピーカー)               │
                                  └── MediaStreamAudioDestination  │
                                          │                        │
                                          └──── MediaRecorder ←── canvas.captureStream(30)
                                                    │
                                                 MP4 / WebM
```

ポイントは **プレビューと書き出しでAnalyserNodeの経路を共有している** こと。書き出し時だけ `MediaStreamAudioDestinationNode` への分岐を追加する設計にすると、「プレビュー用と録画用で音が二重再生される」事故を構造的に防げます。

## 周波数データをかわいいバーにする

### AnalyserNodeの基本

`fftSize: 256` のAnalyserNodeから128個の周波数ビンが取れます。

```js
this.analyser = this.ctx.createAnalyser();
this.analyser.fftSize = 256;
this.analyser.smoothingTimeConstant = 0.8; // バーの動きを滑らかに
```

`smoothingTimeConstant` はイコライザーの「見た目の気持ちよさ」に直結します。0.8前後にすると、バーがカクカクせずぽよんぽよんと動いてくれます。

### 対数スケール割り当て(重要)

周波数ビンを素直に等分してバーに割り当てると、**左端の数本だけが暴れて右側はほぼ無反応**という残念な見た目になります。音楽のエネルギーは低域に偏っているからです。

そこでバーへの割り当てを対数スケールにします。

```js
function getBarValues() {
  player.analyser.getByteFrequencyData(freqData);
  const maxBin = 100; // 高域はほぼ無音なのでカット
  for (let i = 0; i < n; i++) {
    let lo = Math.floor(Math.pow(maxBin, i / n));
    let hi = Math.max(lo + 1, Math.floor(Math.pow(maxBin, (i + 1) / n)));
    let sum = 0;
    for (let b = lo; b < hi; b++) sum += freqData[b];
    const v = sum / (hi - lo) / 255;
    values[i] = Math.pow(v, 1.25); // 低域偏重をさらにならす
  }
  return values;
}
```

- バー `i` が担当するビン範囲を `pow(maxBin, i/n)` で決める → 低域は細かく、高域はまとめて拾う
- 仕上げに `pow(v, 1.25)` で全体のバランスを補正

これだけで「全部のバーがそれっぽく踊る」見た目になります。

## ドット絵モチーフの積み上げエンジン

このアプリの顔である「♡が積み上がるバー」は、**ピクセルパターン定義と描画エンジンを分離**しました。モチーフは0〜3の数値グリッドで定義します。

```js
const PIXEL_PATTERNS = {
  // 0=透明, 1=メイン色, 2=濃い縁取り色, 3=明るいハイライト色
  heart: { size: 8, grid: [
    [0,2,2,0,0,2,2,0],
    [2,1,1,2,2,1,1,2],
    [2,1,3,1,1,1,1,2],
    [2,1,1,1,1,1,1,2],
    [0,2,1,1,1,1,2,0],
    [0,0,2,1,1,2,0,0],
    [0,0,0,2,2,0,0,0],
    [0,0,0,0,0,0,0,0],
  ]},
  sakura: { /* 🌸 */ },
  paw:    { /* 🐾 */ },
  // 新モチーフはここにグリッドを1つ足すだけ
};
```

描画エンジン側は「バー値(0〜1)× 最大段数 = 積む個数」を計算して、下からスプライトを積むだけです。

```js
function drawStackedMotif(ctx, values, area, patternName, colorForLevel) {
  const maxUnits = Math.max(2, Math.floor(area.h / cell));
  ctx.imageSmoothingEnabled = false; // ドットをカリッと
  for (let i = 0; i < n; i++) {
    const units = Math.max(1, Math.round(values[i] * maxUnits));
    for (let k = 0; k < units; k++) {
      const sprite = getPatternSprite(patternName, colorForLevel(k, i), cell);
      ctx.drawImage(sprite, x, area.y + area.h - (k + 1) * cell, cell, cell);
    }
  }
}
```

工夫ポイントが2つあります。

1. **`imageSmoothingEnabled = false`** ─ これがないと拡大時にドットがぼやけて「ドット絵感」が死にます
2. **スプライトのキャッシュ** ─ 毎フレーム全ピクセルを `fillRect` すると重いので、「パターン×色×サイズ」をキーにオフスクリーンCanvasへ事前レンダリングし、本番は `drawImage` 1回にします

```js
const patternCache = new Map();
function getPatternSprite(patternName, mainColor, cellPx) {
  const key = `${patternName}|${mainColor}|${cellPx}`;
  let sprite = patternCache.get(key);
  if (sprite) return sprite;
  // ...オフスクリーンCanvasに1度だけ描いてキャッシュ...
}
```

この構造にしたおかげで、「肉球バーの三毛猫カラー(段ごとに白/茶/黒が切り替わる)」も `colorForLevel` に配列を渡すだけで実現できました。

## フル尺書き出し ─ MediaRecorderの実戦投入

ここが本題です。**動画編集ソフトなしで、Canvasの絵と音声を合成した動画ファイルを作ります。**

### 映像と音声のストリームを合流させる

```js
// 音声: AnalyserNodeの下流に録画用の出口を追加
mediaDest = player.ctx.createMediaStreamDestination();
player.masterGain.connect(mediaDest); // スピーカーへの出力はそのまま維持

// 映像: Canvasを30fpsでキャプチャ
const stream = canvas.captureStream(30);
mediaDest.stream.getAudioTracks().forEach(tr => stream.addTrack(tr));

recorder = new MediaRecorder(stream, {
  mimeType: exportFormat.mime,
  videoBitsPerSecond: 12_000_000,
});
recorder.start(500);  // 500msごとにチャンク回収
await player.play(0); // 曲を頭から再生 → onendedで録画停止
```

MediaRecorderは**リアルタイム録画**なので、書き出し時間=曲の長さです。5分の曲なら5分かかります。オフラインレンダリングはできませんが、その代わり実装は驚くほど素直です。

### コーデックはフォールバックで決める

`video/mp4` を録れるかはブラウザ次第なので、`isTypeSupported()` で上から順に試します。

```js
const MIME_CANDIDATES = [
  { mime: "video/mp4;codecs=avc1.42E01E,mp4a.40.2", ext: "mp4",  label: "MP4 (H.264)" },
  { mime: "video/mp4",                              ext: "mp4",  label: "MP4" },
  { mime: "video/webm;codecs=vp9,opus",             ext: "webm", label: "WebM (VP9)" },
  { mime: "video/webm",                             ext: "webm", label: "WebM" },
];
const format = MIME_CANDIDATES.find(c => MediaRecorder.isTypeSupported(c.mime));
```

どの形式で書き出されるかはUIに明示しておくと親切です(最近のChromeならMP4がそのまま録れます)。

## ハマったポイント集

### 1. AudioContextの自動再生制限

`AudioContext` はユーザー操作なしに作ると `suspended` 状態で止まります。ファイル選択時に生成するのはOK(`decodeAudioData` はsuspendedでも動く)ですが、**再生ボタンのハンドラ内で必ず `resume()`** する必要があります。

```js
async play(offset) {
  if (this.ctx.state === "suspended") await this.ctx.resume();
  // ...
}
```

### 2. `onended` が非同期に飛んでくる罠(今回一番のバグ)

シークは「今のソースを止めて、新しいオフセットでソースを作り直す」実装になります。このとき `source.stop()` の `onended` イベントは **同期的には発火せず、次のイベントループで飛んできます**。

最初は「手動停止フラグを立てる → stop() → フラグを戻す」という共有フラグで実装していたのですが、フラグを戻した**後**に旧ソースの `onended` が遅れて発火し、新しい再生状態を破壊するバグを踏みました(再生中にシークすると止まる)。

対策は、フラグを共有変数ではなく**ソースオブジェクト自身に持たせる**ことです。

```js
stopSource() {
  if (this.source) {
    this.source._manualStop = true; // このソース個体だけに印を付ける
    this.source.stop();
  }
}
// onended側
src.onended = () => {
  if (src._manualStop) return; // 手動停止なら何もしない
  // 自然に曲が終わった時だけ、ここで録画停止などを行う
};
```

`AudioBufferSourceNode` は使い捨て(1回しかstartできない)なので、状態もソース個体に紐付けるのが正解でした。

### 3. バックグラウンドタブでフレームが間引かれる

`requestAnimationFrame` はタブが非表示になると止まり、`captureStream` の映像フレームも供給されなくなります。**書き出し中にタブを最小化すると、音は正常なのに映像がカクカク(または静止)の動画ができあがります。**

アプリ側でできる対策は限られるので、

- 書き出し中は「タブを閉じたり最小化しないでください」と明示する
- `beforeunload` で誤タブ閉じをガードする

```js
window.addEventListener("beforeunload", e => {
  if (state.exporting) { e.preventDefault(); e.returnValue = ""; }
});
```

あたりが現実解です。

### 4. 大きい画像でメモリが溶ける

スマホで撮った4000px超の画像を10枚読むと、それだけで数百MBです。アップロード時点でオフスクリーンCanvasに出力解像度程度(1920×1080)へ縮小してから保持するようにしました。元画像はその場で `URL.revokeObjectURL()` して手放します。

## スライドショーの切り替え計算

「曲の長さ ÷ 画像枚数」で1枚あたりの表示時間を決め、境界の手前0.7秒だけクロスフェードします。現在時刻 `t` から純粋関数で状態を導出する形にしておくと、シークしても壊れないのが地味に効きます。

```js
function getSlideState(t) {
  const per = duration / imageCount;      // 1枚あたりの秒数
  const raw = Math.floor(t / per);        // 今何枚目か
  const tIn = t - raw * per;              // その画像に入ってからの経過
  let mix = 0;
  if (tIn > per - transDur) mix = (tIn - (per - transDur)) / transDur;
  return { a: raw % n, b: (raw + 1) % n, mix };
}
```

## おわりに

- ブラウザAPIだけで「音に反応するMVのフル尺書き出し」は普通に実用になる
- AnalyserNodeは**対数スケール割り当て**で見た目が激変する
- ドット絵モチーフは「数値グリッド定義 + 共通積み上げエンジン」に分離すると追加が一瞬
- MediaRecorderはリアルタイム録画という制約さえ飲めば、素直で強力

「知り合いが作ってた30秒のやつ、フル尺で自分でも作れるのでは?」という思いつきが、1枚のHTMLで実現できるのは良い時代だなと思います。Canvas 2DとWeb Audioの組み合わせは題材としてもかなり楽しいので、ぜひ自分の「かわいい」を積み上げてみてください🎀
