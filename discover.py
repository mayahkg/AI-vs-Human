"""
discover.py (v3) - find candidate human-made and AI-made short drama videos on YouTube.

Usage (from your project folder, with (.venv) active):
    python discover.py

What's new in v3:
  * more NON-fruit AI queries, so the AI side isn't only fruit dramas
  * AI videos are split into ai_fruit_candidate and ai_other_candidate
  * a rough theme tag (cheating, revenge, family, ...) so you can match themes across groups
  * re-running MERGES into data/candidates.csv instead of overwriting it, so your
    transcription progress (status, language, n_words, ...) is kept.
    A backup is saved to data/candidates_backup.csv each run.

Search results are cached in data/, so only NEW queries spend quota.
"""
import os
import re
import json
import shutil
import hashlib
import pathlib

import isodate
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If a local .env file exists (and python-dotenv is installed), load it. Never type the key here.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# "YT_API_KEY" is the NAME of the variable, not the key. Leave this line exactly as is.
yt = build("youtube", "v3", developerKey=os.environ["YT_API_KEY"])

OUT = pathlib.Path("data")
OUT.mkdir(exist_ok=True)
CSV = OUT / "candidates.csv"

# ---------------- settings you can edit ----------------
HUMAN_QUERIES = [
    "drama skit",
    "family betrayal drama scene",
    "pov drama skit actress",
    "cheating boyfriend skit acting",
    "family drama skit acting",
    "mini drama skit original",
    "short film drama emotional scene",
]
AI_QUERIES = [                       # (mostly fruit; these are already cached)
    "ai fruit drama",
    "fruit love island ai",
    "ai mini drama",
    "ai generated drama story",
    "ai fruit cheating drama",
    "ai fruit story episode",
    "ai cinema fruit",
    "ai animal drama",
    "ai baby drama",
    "ai love story episode english",
    "ai generated story drama",
    "ai drama episode",
]
AI_QUERIES_OTHER = [                 # NEW: aim away from fruit
    "veo 3 drama scene",
    "ai generated realistic drama people",
    "ai cheating story drama",
    "ai revenge story",
    "ai mother in law drama",
    "ai family drama story",
    "ai office drama story",
    "ai husband wife argument",
    "ai romance short film",
    "ai cartoon drama story",
    "sora short story drama",
    "ai horror short story",
]
HUMAN_CHARACTER_QUERIES = [          # NEW: human-made dramas with non-live-action characters
    "stop motion drama story",
    "claymation drama skit",
    "puppet drama skit",
    "barbie doll drama skit",
    "toy drama story skit",
    "sylvanian families drama",
    "animated food characters drama",
    "fruit stop motion story",
    "talking fruit puppet skit",
    "2d animated drama story voice acted",
    "gacha life drama story",
    "sims 4 drama story",
    "roblox drama story",
]
# Set True to also search recent (post-2022) human-character videos. They are only kept if the
# text mentions a hand-made process (stop motion, puppets, ...), because recent unlabeled
# animation could be AI. Doubles the quota cost for these queries.
INCLUDE_RECENT_HUMAN_CHAR = False
PAGES = 2                              # 50 results per page; each page costs 100 quota units
HUMAN_BEFORE = "2023-01-01T00:00:00Z"  # human candidates: posted before this date
AI_AFTER = "2025-01-01T00:00:00Z"      # AI candidates: posted after this date
MIN_SEC, MAX_SEC = 15, 90              # keep short-form lengths only
MAX_PER_CHANNEL = 4                    # stop one creator/series from flooding the data

AI_TERMS = [
    "ai generated", "ai-generated", "made with ai", "aivideo", "aigenerated",
    "veo", "sora", "kling", "fruit drama", "aicrashout", "aifruit", "aifruitdrama",
    "aicinema", "aidrama", "aidorama", "aistory", "fruitloveisland",
    "ai drama", "ai story",
]
HUMAN_BLOCK = [
    "kdrama", "k-drama", "cdrama", "c-drama", "kmovie", "kmovies", "korean drama",
    "chinese drama", "dorama", "manhwa", "minecraft", "reddit", "askreddit", "tts",
    "podcast", "true crime", "stream", "musical", "trailer", "movie", "netflix",
    "sopranos", "breaking bad", "game of thrones", "spongebob", "joke", "jokes", "status",
]
AI_BLOCK = [
    "art generator", "image generator", "tutorial", "how to", "prompt",
    "free ai", "app", "tool", "review",
]
# anything matching this counts as a "fruit" video (substring 'fruit' covers #aifruit, fruitstory...)
FRUIT_RE = re.compile(
    r"fruit|\b(banana|apple|mango|strawberry|watermelon|orange|grape|pineapple|peach|"
    r"cherry|lemon|avocado|coconut|kiwi|pear)s?\b")
# rough themes, checked in this order; first match wins
THEMES = {
    "cheating_betrayal": ["cheat", "betray", "affair", "infidel", "caught"],
    "revenge": ["revenge", "payback", "karma"],
    "family": ["mom", "mother", "dad", "father", "family", "mil", "in-law",
               "baby", "sister", "brother", "parent"],
    "romance": ["love", "date", "crush", "romance", "boyfriend", "girlfriend",
                "husband", "wife"],
    "workplace": ["boss", "office", "coworker", "job"],
    "horror": ["horror", "scary", "ghost", "haunted"],
}
# --------------------------------------------------------


def make_regex(terms):
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b")


AI_RE, HUMAN_BLOCK_RE, AI_BLOCK_RE = (
    make_regex(AI_TERMS), make_regex(HUMAN_BLOCK), make_regex(AI_BLOCK))

# off-genre words for the character group (tutorials, ASMR, gameplay, etc.)
CHAR_BLOCK = ["tutorial", "how to", "asmr", "review", "trailer", "unboxing",
              "speedrun", "walkthrough", "gameplay", "craft", "diy", "slime"]
# words showing a hand-made process, used to accept RECENT human-character videos
HUMAN_PROCESS = ["stop motion", "stop-motion", "claymation", "puppet", "puppets", "handmade",
                 "hand drawn", "hand-drawn", "gacha", "sims 4", "roblox", "barbie", "doll",
                 "dolls", "lego", "plush", "plushie"]
CHAR_BLOCK_RE, HUMAN_PROCESS_RE = make_regex(CHAR_BLOCK), make_regex(HUMAN_PROCESS)


def theme_of(t):
    for name, words in THEMES.items():
        if any(re.search(r"\b" + re.escape(w), t) for w in words):
            return name
    return "other"


def cache_path(q, filters):
    """Each search is saved to disk so rerunning it costs no quota."""
    key = json.dumps([q, filters], sort_keys=True)
    return OUT / f"cache_{hashlib.md5(key.encode()).hexdigest()[:10]}.json"


def search_ids(q, **filters):
    path = cache_path(q, filters)
    if path.exists():
        return json.loads(path.read_text())
    ids, token = [], None
    for _ in range(PAGES):
        resp = yt.search().list(
            q=q, part="id", type="video", videoDuration="short",
            maxResults=50, pageToken=token, relevanceLanguage="en", **filters,
        ).execute()
        ids += [it["id"]["videoId"] for it in resp["items"]]
        token = resp.get("nextPageToken")
        if not token:
            break
    path.write_text(json.dumps(ids))
    return ids


def get_details(ids):
    rows = []
    for k in range(0, len(ids), 50):
        resp = yt.videos().list(
            part="snippet,contentDetails,statistics,status",
            id=",".join(ids[k:k + 50]),
        ).execute()
        for v in resp["items"]:
            s = v["snippet"]
            rows.append({
                "video_id": v["id"],
                "url": f"https://www.youtube.com/shorts/{v['id']}",
                "channel": s["channelId"],
                "title": s["title"],
                "description": s.get("description", ""),
                "tags": " ".join(s.get("tags", [])),
                "published": s["publishedAt"],
                "audio_lang": s.get("defaultAudioLanguage") or s.get("defaultLanguage"),
                "duration": isodate.parse_duration(
                    v["contentDetails"]["duration"]).total_seconds(),
                "views": v.get("statistics", {}).get("viewCount"),
                "synthetic_flag": v.get("status", {}).get("containsSyntheticMedia"),
            })
    return rows


# ---------------- collect ----------------
all_rows = []
try:
    for q in HUMAN_QUERIES:
        for r in get_details(search_ids(q, publishedBefore=HUMAN_BEFORE)):
            r["source_query"], r["query_type"] = q, "human_q"
            all_rows.append(r)
    for q in AI_QUERIES + AI_QUERIES_OTHER:
        for r in get_details(search_ids(q, publishedAfter=AI_AFTER)):
            r["source_query"], r["query_type"] = q, "ai_q"
            all_rows.append(r)
    for q in HUMAN_CHARACTER_QUERIES:
        for r in get_details(search_ids(q, publishedBefore=HUMAN_BEFORE)):
            r["source_query"], r["query_type"] = q, "human_char_q"
            all_rows.append(r)
        if INCLUDE_RECENT_HUMAN_CHAR:
            for r in get_details(search_ids(q, publishedAfter=HUMAN_BEFORE)):
                r["source_query"], r["query_type"] = q, "human_char_q"
                all_rows.append(r)
except HttpError as e:
    print("API error:", e)   # e.g. quota exceeded; cached searches are kept, rerun later

if not all_rows:
    raise SystemExit("No results collected. Check the error above (key, API enabled, quota).")

# ---------------- clean and label ----------------
df = pd.DataFrame(all_rows).drop_duplicates("video_id")
df = df[(df.duration >= MIN_SEC) & (df.duration <= MAX_SEC)].copy()
df = df[df.audio_lang.isna() | df.audio_lang.astype(str).str.lower().str.startswith("en")].copy()
df["text"] = (df.title + " " + df.description + " " + df.tags).str.lower()


def label(r):
    flagged = pd.notna(r.synthetic_flag) and bool(r.synthetic_flag)
    if (AI_RE.search(r.text) or flagged) and r.published >= AI_AFTER:
        return "uncertain" if AI_BLOCK_RE.search(r.text) else "ai_candidate"
    if r.query_type == "human_q" and r.published < "2023-01-01":
        return "uncertain" if HUMAN_BLOCK_RE.search(r.text) else "human_candidate"
    if r.query_type == "human_char_q":
        if CHAR_BLOCK_RE.search(r.text):
            return "uncertain"
        if r.published < HUMAN_BEFORE or HUMAN_PROCESS_RE.search(r.text):
            return "human_character_candidate"
    return "uncertain"


df["group"] = df.apply(label, axis=1)
df = df[df.group != "uncertain"]
if df.empty:
    raise SystemExit("Nothing left after filtering. Try more queries or looser settings.")

df = (df.sample(frac=1, random_state=42)
        .groupby(["group", "channel"]).head(MAX_PER_CHANNEL)
        .drop(columns=["text"]))
df["status"] = "new"

# ---------------- merge with earlier results (keeps transcription progress) ----------------
if CSV.exists():
    shutil.copy(CSV, OUT / "candidates_backup.csv")
    old = pd.read_csv(CSV)
    fresh = df[~df.video_id.isin(old.video_id)]
    print(f"Existing rows kept: {len(old)} | new rows added: {len(fresh)}")
    df = pd.concat([old, fresh], ignore_index=True)

# ---------------- split AI into fruit / other, add themes ----------------
text = (df.title.fillna("") + " " + df.description.fillna("") + " " +
        df.tags.fillna("")).str.lower()
is_ai = df.group.str.startswith("ai_")
is_fruit = text.apply(lambda t: bool(FRUIT_RE.search(t)))
df.loc[is_ai, "group"] = np.where(is_fruit[is_ai], "ai_fruit_candidate", "ai_other_candidate")
df["theme"] = text.apply(theme_of)
df["is_fruit"] = is_fruit                      # lets you find human-made fruit "twins" too
df["label_basis"] = np.where(
    df.group.str.startswith("human"),
    np.where(df.published < HUMAN_BEFORE, "pre2023", "process_terms"),
    "ai_tag")

df.to_csv(CSV, index=False)

review = (df.groupby("group").head(30)[["video_id", "group", "title", "url"]].assign(keep=""))
review.to_csv(OUT / "review_sample.csv", index=False)

# ---------------- summary ----------------
print("\nVideos per group:")
print(df.group.value_counts())
print("\nChannels per group:")
print(df.groupby("group").channel.nunique())
print("\nThemes by group:")
print(pd.crosstab(df.theme, df.group))
print("\nStatus by group (transcription progress):")
print(pd.crosstab(df.group, df.status))
print("\nSaved to", CSV, "(backup:", OUT / "candidates_backup.csv)")