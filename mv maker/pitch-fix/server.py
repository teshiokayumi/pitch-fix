"""ローカル専用の補正サーバー。

    python server.py

ブラウザ側で「ガイドを再生しながら録音」し、その WAV をここへ送る。
補正は重いのでバックグラウンドスレッドで走らせ、進捗をポーリングで返す。
音声はこの PC から出ない(127.0.0.1 にしか bind しない)。
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pitchfix.params import CorrectionParams
from pitchfix.pipeline import correct_take

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_MB = 400
MAX_JOBS_KEPT = 12


@dataclass
class Job:
    id: str
    status: str = "queued"     # queued / running / done / error
    progress: float = 0.0
    message: str = "順番待ち"
    error: str | None = None
    summary: dict | None = None
    curves: dict | None = None
    audio_path: Path | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "summary": self.summary,
        }


class JobStore:
    """ジョブの保管。古いものから捨てて、ディスクとメモリを溜め込まない。"""

    def __init__(self, work_dir: Path):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.work_dir = work_dir

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="そのジョブは見つかりません")
        return job

    def _evict_locked(self) -> None:
        if len(self._jobs) <= MAX_JOBS_KEPT:
            return
        stale = sorted(self._jobs.values(), key=lambda j: j.created)
        for job in stale[: len(self._jobs) - MAX_JOBS_KEPT]:
            if job.status in ("queued", "running"):
                continue
            self._jobs.pop(job.id, None)
            if job.audio_path and job.audio_path.exists():
                job.audio_path.unlink(missing_ok=True)


work_dir = Path(tempfile.mkdtemp(prefix="pitchfix_"))
store = JobStore(work_dir)
executor = ThreadPoolExecutor(max_workers=1)  # 重い処理を同時に走らせない


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    executor.shutdown(wait=False, cancel_futures=True)
    shutil.rmtree(work_dir, ignore_errors=True)


app = FastAPI(title="pitch-fix", lifespan=lifespan)


def _decode_upload(raw: bytes, label: str) -> tuple[np.ndarray, int]:
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label}が空です")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"{label}が大きすぎます({MAX_UPLOAD_MB}MB まで)。曲を短く区切ってください。",
        )
    try:
        data, fs = sf.read(io.BytesIO(raw), dtype="float64", always_2d=True)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"{label}を読み込めませんでした: {exc}") from exc
    return np.ascontiguousarray(data.mean(axis=1)), int(fs)


def _run_job(job: Job, guide, take, params: CorrectionParams) -> None:
    job.status = "running"
    job.message = "解析を開始しています"

    def progress(fraction: float, message: str) -> None:
        job.progress = fraction
        job.message = message

    try:
        result = correct_take(guide, take, params, progress=progress)
        out_path = store.work_dir / f"{job.id}.wav"
        result.save(out_path)
        job.audio_path = out_path
        job.summary = result.summary()
        job.curves = result.curves
        job.progress = 1.0
        job.message = "完了"
        job.status = "done"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc) or exc.__class__.__name__
        job.message = "エラー"


@app.post("/api/jobs")
async def create_job(
    guide: UploadFile = File(...),
    take: UploadFile = File(...),
    params: str = Form("{}"),
):
    try:
        parsed = json.loads(params) if params else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="パラメータの形式が不正です")

    correction = CorrectionParams.from_dict(parsed).validated()
    # await で読むことで、大きなアップロード中も進捗ポーリングが止まらない
    guide_audio = _decode_upload(await guide.read(), "ガイド音源")
    take_audio = _decode_upload(await take.read(), "録音")

    job = store.create()
    executor.submit(_run_job, job, guide_audio, take_audio, correction)
    return {"id": job.id}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    return store.get(job_id).public()


@app.get("/api/jobs/{job_id}/audio")
async def job_audio(job_id: str):
    job = store.get(job_id)
    if job.status != "done" or not job.audio_path or not job.audio_path.exists():
        raise HTTPException(status_code=409, detail="まだ完成していません")
    return FileResponse(job.audio_path, media_type="audio/wav",
                        filename=f"pitchfix_{job.id}.wav")


@app.get("/api/jobs/{job_id}/curves")
async def job_curves(job_id: str):
    job = store.get(job_id)
    if job.status != "done" or job.curves is None:
        raise HTTPException(status_code=409, detail="まだ完成していません")
    return JSONResponse(job.curves)


@app.get("/api/defaults")
async def defaults():
    return CorrectionParams().to_dict()


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="pitch-fix のローカルサーバーを起動します")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  pitch-fix を起動しました → {url}")
    print("  ヘッドホンを着けてから録音してください。終了するには Ctrl+C。\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
