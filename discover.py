"""
discover.py - find candidate human-made and AI-made short drama videos on YouTube.
 
Usage (from your project folder, with (.venv) active and YT_API_KEY set):
    python discover.py
 
Output: data/candidates.csv (plus cache files so reruns don't spend quota again)
"""
import os
import re
import json
import hashlib
import pathlib
 
import isodate
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
 
# The key is read from your terminal's environment, never typed into this file.
yt = build("youtube", "v3", developerKey=os.environ["YT_API_KEY"])
 
OUT = pathlib.Path("data")
OUT.mkdir(exist_ok=True)
 
# ---------------- settings you can edit ----------------
HUMAN_QUERIES = [
    "drama skit",
    "reddit story reenactment",
    "cheating storytime skit",
    "family betrayal drama scene",
    "revenge story skit",
]
AI_QUERIES = [
    "ai fruit drama",
    "fruit love island ai",
    "ai mini drama",
    "ai generated drama story",
]
PAGES = 2                              # 50 results per page; each page costs 100 quota units
HUMAN_BEFORE = "2023-01-01T00:00:00Z"  # human candidates: posted before this date
AI_AFTER = "2025-01-01T00:00:00Z"      # AI candidates: posted after this date
MIN_SEC, MAX_SEC = 15, 90              # keep short-form lengths only
MAX_PER_CHANNEL = 4                    # stop one creator/series from flooding the data
AI_TERMS = [
    "ai generated", "ai-generated", "made with ai", "aivideo", "aigenerated",
    "veo", "sora", "kling", "fruit drama", "aicrashout",
]
# --------------------------------------------------------
 
AI_REGEX = re.compile(r"\b(" + "|".join(re.escape(t) for t in AI_TERMS) + r")\b")
 
 
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
            q=q,
            part="id",
            type="video",
            videoDuration="short",      # YouTube's "under 4 minutes" bucket
            maxResults=50,
            pageToken=token,
            relevanceLanguage="en",
            **filters,
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
                "duration": isodate.parse_duration(
                    v["contentDetails"]["duration"]).total_seconds(),
                "views": v.get("statistics", {}).get("viewCount"),
                # creator's own "altered or synthetic content" disclosure, if the API returns it
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
    for q in AI_QUERIES:
        for r in get_details(search_ids(q, publishedAfter=AI_AFTER)):
            r["source_query"], r["query_type"] = q, "ai_q"
            all_rows.append(r)
except HttpError as e:
    # e.g. quota exceeded. Searches already cached are kept; rerun later to continue.
    print("API error:", e)
 
if not all_rows:
    raise SystemExit("No results collected. Check the error above (key, API enabled, quota).")
 
# ---------------- clean and label ----------------
df = pd.DataFrame(all_rows).drop_duplicates("video_id")
df = df[(df.duration >= MIN_SEC) & (df.duration <= MAX_SEC)].copy()
 
text = (df.title + " " + df.description + " " + df.tags).str.lower()
df["ai_keyword"] = text.apply(lambda t: bool(AI_REGEX.search(t)))
 
 
def label(r):
    flagged = pd.notna(r.synthetic_flag) and bool(r.synthetic_flag)
    if r.ai_keyword or flagged:
        return "ai_candidate"
    if r.query_type == "human_q" and r.published < "2023-01-01":
        return "human_candidate"
    return "uncertain"
 
 
df["group"] = df.apply(label, axis=1)
df = df[df.group != "uncertain"]
 
if df.empty:
    raise SystemExit("Nothing left after filtering. Try more queries or looser settings.")
 
# shuffle, then cap videos per channel so no single creator dominates
df = (
    df.sample(frac=1, random_state=42)
      .groupby(["group", "channel"])
      .head(MAX_PER_CHANNEL)
)
 
df["status"] = "new"
df.to_csv(OUT / "candidates.csv", index=False)
 
# ---------------- summary ----------------
print("\nVideos per group:")
print(df.group.value_counts())
print("\nChannels per group:")
print(df.groupby("group").channel.nunique())
print("\nRows where synthetic_flag was filled in:", int(df.synthetic_flag.notna().sum()))
print("Saved to", OUT / "candidates.csv")
 