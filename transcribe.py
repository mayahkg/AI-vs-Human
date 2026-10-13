"""
transcribe.py - download audio for candidate videos and transcribe it with Whisper.

Usage (from your project folder, with (.venv) active):
    pip install yt-dlp faster-whisper
    python transcribe.py

It reads data/candidates.csv, processes rows whose status is "new", and writes:
    transcripts/<video_id>.txt     the transcript text
    data/candidates.csv            updated after EVERY video (so you can stop and resume)
Audio files are deleted right after transcription; only the text is kept.

Start small: LIMIT_PER_GROUP = 10 is a test run. Raise it once the test looks good.
"""
import sys
import time
import random
import pathlib
import subprocess

import pandas as pd
from faster_whisper import WhisperModel

# ---------------- settings you can edit ----------------
CSV = pathlib.Path("data/candidates.csv")
AUDIO_DIR = pathlib.Path("audio")
TRANSCRIPT_DIR = pathlib.Path("transcripts")
LIMIT_PER_GROUP = 10        # how many NEW videos to process per group in this run
MODEL_SIZE = "small"        # "base" is faster but less accurate; "medium" is slower and better
MIN_WORDS = 30              # shorter transcripts are marked too_short
MIN_LANG_PROB = 0.7         # language must be English with at least this confidence
DELAY_RANGE = (3, 8)        # random pause (seconds) between downloads, to be polite
# --------------------------------------------------------

AUDIO_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV)
for col, default in [("language", ""), ("lang_prob", float("nan")),
                     ("n_words", float("nan")), ("note", "")]:
    if col not in df.columns:
        df[col] = default
df["language"] = df["language"].astype("object")
df["note"] = df["note"].astype("object")

todo = (df[df.status == "new"]
        .groupby("group", group_keys=False)
        .head(LIMIT_PER_GROUP)
        .index)
print(f"{len(todo)} videos to process this run")

print(f"Loading Whisper model '{MODEL_SIZE}' (the first run downloads it)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def download_audio(video_id, url):
    """Download the best audio-only stream. No ffmpeg install needed; Whisper decodes it."""
    for old in AUDIO_DIR.glob(f"{video_id}.*"):
        old.unlink()
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "--no-playlist",
         "-o", str(AUDIO_DIR / f"{video_id}.%(ext)s"), url],
        capture_output=True, text=True)
    files = list(AUDIO_DIR.glob(f"{video_id}.*"))
    if result.returncode != 0 or not files:
        raise RuntimeError(result.stderr.strip().splitlines()[-1][:150]
                           if result.stderr.strip() else "download failed")
    return files[0]


for n, i in enumerate(todo, 1):
    vid, url = df.at[i, "video_id"], df.at[i, "url"]
    audio = None
    try:
        audio = download_audio(vid, url)
        segments, info = model.transcribe(str(audio), language=None, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        n_words = len(text.split())

        df.at[i, "language"] = info.language
        df.at[i, "lang_prob"] = round(info.language_probability, 3)
        df.at[i, "n_words"] = n_words

        if info.language != "en" or info.language_probability < MIN_LANG_PROB:
            df.at[i, "status"] = "non_english"
        elif n_words < MIN_WORDS:
            df.at[i, "status"] = "too_short"
        else:
            (TRANSCRIPT_DIR / f"{vid}.txt").write_text(text)
            df.at[i, "status"] = "transcribed"
    except Exception as e:
        df.at[i, "status"] = "failed"
        df.at[i, "note"] = str(e)[:150]
    finally:
        if audio is not None and pathlib.Path(audio).exists():
            pathlib.Path(audio).unlink()          # keep the text, not the audio
    df.to_csv(CSV, index=False)                    # save after every video
    print(f"[{n}/{len(todo)}] {vid} -> {df.at[i, 'status']}")
    time.sleep(random.uniform(*DELAY_RANGE))

print("\nStatus counts by group:")
print(df.groupby(["group", "status"]).size().to_string())
