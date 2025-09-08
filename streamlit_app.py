# streamlit_app.py — TV Scheduler (ChatGPT build, based on work by Márk Németh)
from __future__ import annotations
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

Audience = str  # "Kids" | "Teen" | "Adults"

@dataclass
class Movie:
    title: str
    genre: str
    duration_sec: int
    target_audience: Audience

@dataclass
class Ad:
    ad_title: str
    category: str
    duration_sec: int
    target: Audience
    importance: str  # "priority" | "mid"
    at_least_times_played: int
    revenue_on_ad: float

@dataclass
class Slot:
    start_time: datetime
    end_time: datetime
    content_type: str  # "movie" | "ad_break" | "ad"
    title: str
    audience: Audience
    importance: str | None = None
    revenue: float = 0.0

# --- Tables ---
TIME_TABLE: Dict[int, Dict[Audience, float]] = {}
for h in range(8, 12): TIME_TABLE[h] = {"Kids": 1.0, "Teen": 0.9, "Adults": 0.8}
for h in range(12, 16): TIME_TABLE[h] = {"Kids": 0.9, "Teen": 1.0, "Adults": 0.8}
for h in range(16, 18): TIME_TABLE[h] = {"Kids": 0.8, "Teen": 1.0, "Adults": 0.9}
for h in range(18, 23): TIME_TABLE[h] = {"Kids": 0.8, "Teen": 0.9, "Adults": 1.0}

GENRE_TABLE = {
    "Action": {"Kids": 0.08, "Teen": 0.10, "Adults": 0.10},
    "Comedy": {"Kids": 0.08, "Teen": 0.10, "Adults": 0.10},
    "Thriller": {"Kids": 0.08, "Teen": 0.09, "Adults": 0.10},
    "Kids": {"Kids": 0.10, "Teen": 0.09, "Adults": 0.08},
}
MID_ROLL_TABLE = {
    "Kids": {"Kids": 0.10, "Teen": 0.09, "Adults": 0.08},
    "Teen": {"Kids": 0.09, "Teen": 0.10, "Adults": 0.08},
    "Adults": {"Kids": 0.08, "Teen": 0.09, "Adults": 0.10},
}
BETWEEN_TABLE = {
    "Kids|Kids": {"Kids": 0.10, "Teen": 0.09, "Adults": 0.08},
    "Kids|Teen": {"Kids": 0.10, "Teen": 0.10, "Adults": 0.09},
    "Kids|Adults": {"Kids": 0.10, "Teen": 0.10, "Adults": 0.10},
    "Teen|Kids": {"Kids": 0.10, "Teen": 0.10, "Adults": 0.09},
    "Teen|Teen": {"Kids": 0.09, "Teen": 0.10, "Adults": 0.09},
    "Teen|Adults": {"Kids": 0.09, "Teen": 0.10, "Adults": 0.10},
    "Adults|Kids": {"Kids": 0.10, "Teen": 0.10, "Adults": 0.10},
    "Adults|Teen": {"Kids": 0.09, "Teen": 0.10, "Adults": 0.10},
    "Adults|Adults": {"Kids": 0.08, "Teen": 0.09, "Adults": 0.10},
}

# --- Helpers ---
def split_into_parts(duration: int) -> List[int]:
    b = duration // 3
    parts = [b, b, b]
    for i in range(duration - sum(parts)):
        parts[i % 3] += 1
    return parts

def today_at(h: int, m: int = 0, s: int = 0) -> datetime:
    now = datetime.now()
    return now.replace(hour=h, minute=m, second=s, microsecond=0)

def rps(ad: Ad) -> float:
    return ad.revenue_on_ad / max(1, ad.duration_sec)

# --- Core logic ---
def schedule_all_movies(movies: List[Movie]) -> tuple[list[Slot], list[Movie]]:
    schedule, pool, remaining = [], movies.copy(), movies.copy()
    current, end_day = today_at(8), today_at(23)
    while pool and current < end_day:
        scored = [(round(TIME_TABLE.get(current.hour, {}).get(m.target_audience, 0)
                         + GENRE_TABLE.get(m.genre, {}).get(m.target_audience, 0), 4), m)
                  for m in pool]
        scored.sort(key=lambda x: (-x[0], x[1].duration_sec))
        best = scored[0][1]
        for i, dur in enumerate(split_into_parts(best.duration_sec)):
            end = current + timedelta(seconds=dur)
            schedule.append(Slot(current, end, "movie", f"{best.title} (Part {i+1}/3)", best.target_audience))
            current = end
            if i < 2:
                br_end = current + timedelta(seconds=360)
                schedule.append(Slot(current, br_end, "ad_break", "MID-ROLL", best.target_audience))
                current = br_end
        pool.remove(best)
        remaining = [m for m in remaining if m.title != best.title]
        if pool and current < end_day:
            br_end = current + timedelta(seconds=900)
            schedule.append(Slot(current, br_end, "ad_break", "BETWEEN-MOVIE", best.target_audience))
            current = br_end
    return schedule, remaining

def fill_ad_breaks(schedule_in: List[Slot], ads: List[Ad]) -> List[Slot]:
    usage: Dict[str, int] = {}
    final, sched = [], [Slot(**s.__dict__) for s in schedule_in]
    i = 0
    while i < len(sched):
        slot = sched[i]
        if slot.content_type != "ad_break":
            final.append(slot); i += 1; continue
        prev_movie = next((s for s in reversed(final) if s.content_type == "movie"), None)
        prev_aud = prev_movie.audience if prev_movie else slot.audience
        next_movie = next((s for s in sched[i+1:] if s.content_type == "movie"), None)
        next_aud = next_movie.audience if next_movie else prev_aud
        left = int((slot.end_time - slot.start_time).total_seconds())
        cur_t, last_cat, used = slot.start_time, None, set()
        while True:
            cands = []
            for ad in ads:
                if ad.duration_sec > left or ad.category == last_cat or ad.ad_title in used: continue
                played = usage.get(ad.ad_title, 0)
                eff = "mid" if (ad.importance == "priority" and played >= ad.at_least_times_played) else ad.importance
                imp_sc = 1.0 if eff == "priority" else 0.9
                table = MID_ROLL_TABLE[prev_aud] if slot.title == "MID-ROLL" else BETWEEN_TABLE[f"{prev_aud}|{next_aud}"]
                base = table.get(ad.target, 0.0)
                cands.append((base, imp_sc, rps(ad), ad, eff))
            if not cands: break
            cands.sort(key=lambda x: (-x[0], -x[1], -x[2]))
            _, _, _, pick, eff = cands[0]
            end_t = cur_t + timedelta(seconds=pick.duration_sec)
            final.append(Slot(cur_t, end_t, "ad", pick.ad_title, pick.target, eff, pick.revenue_on_ad))
            used.add(pick.ad_title); last_cat = pick.category; cur_t = end_t; left -= pick.duration_sec
            if eff == "priority": usage[pick.ad_title] = usage.get(pick.ad_title, 0) + 1
        shift = int((cur_t - slot.start_time).total_seconds()) - int((slot.end_time - slot.start_time).total_seconds())
        if shift != 0:
            for j in range(i+1, len(sched)):
                sched[j].start_time += timedelta(seconds=shift)
                sched[j].end_time += timedelta(seconds=shift)
        i += 1
    return final

# --- IO helpers ---
def rows_to_movies(df: pd.DataFrame) -> List[Movie]:
    if df.empty: return []
    def g(r, *names): 
        for n in names:
            if n in df.columns: return r[n]
        return None
    movies = []
    for _, r in df.iterrows():
        title = g(r, "title", "TITLE"); genre = g(r, "genre", "main genre")
        dur = g(r, "duration_sec", "duration (sec)"); ta = g(r, "target_audience")
        if pd.isna(title) or pd.isna(genre) or pd.isna(dur) or pd.isna(ta): continue
        movies.append(Movie(str(title), str(genre), int(dur), str(ta)))
    return movies

def rows_to_ads(df: pd.DataFrame) -> List[Ad]:
    if df.empty: return []
    def g(r, *names):
        for n in names:
            if n in df.columns: return r[n]
        return None
    ads = []
    for _, r in df.iterrows():
        ad_title = g(r, "ad_title"); category = g(r, "category")
        dur = g(r, "duration_sec", "duration (sec)"); target = g(r, "target")
        importance = g(r, "importance") or "mid"
        minplays = g(r, "at_least_times_played") or 0; revenue = g(r, "revenue_on_ad") or 0
        if pd.isna(ad_title) or pd.isna(category) or pd.isna(dur) or pd.isna(target): continue
        ads.append(Ad(str(ad_title), str(category), int(dur), str(target), str(importance), int(minplays), float(revenue)))
    return ads

def schedule_to_df(slots: List[Slot]) -> pd.DataFrame:
    rows = []
    for s in slots:
        revps = (s.revenue / max(1, int((s.end_time - s.start_time).total_seconds()))) if s.content_type == "ad" else None
        rows.append({
            "start": s.start_time.strftime("%H:%M:%S"),
            "end": s.end_time.strftime("%H:%M:%S"),
            "title": s.title,
            "type": s.content_type,
            "audience": s.audience,
            "importance": s.importance or "-",
            "rev_per_sec": None if revps is None else round(revps, 2),
        })
    return pd.DataFrame(rows)

def analyze_ad(slots: List[Slot], ad: Ad) -> Dict[str, float | int]:
    total = sum(1 for s in slots if s.content_type == "ad" and s.title == ad.ad_title)
    req = ad.at_least_times_played
    quota = (total / req * 100) if req > 0 else 0.0
    match = 0
    for i, s in enumerate(slots):
        if s.content_type != "ad" or s.title != ad.ad_title: continue
        prev = next((x for x in reversed(slots[:i]) if x.content_type == "movie"), None)
        nextm = next((x for x in slots[i+1:] if x.content_type == "movie"), None)
        ctx = {prev.audience} if prev and s.title == "MID-ROLL" else {prev.audience if prev else None, nextm.audience if nextm else None}
        if ad.target in ctx: match += 1
    return {"total_plays": total, "required": req, "pct_quota": round(quota, 1), "match_plays": match, "pct_match": round((match/total*100) if total else 0.0, 1)}

# --- UI ---
st.set_page_config(page_title="TV Scheduler (ChatGPT × Márk Németh)", layout="wide")
st.title("TV Scheduler")
st.caption("Made by ChatGPT, based on the work by Márk Németh")

with st.sidebar:
    st.header("Load data")
    use_demo = st.toggle("Use small demo data", value=False)
    mv_file = st.file_uploader("movie_dataset.xlsx", type=["xlsx", "xls"])
    ad_file = st.file_uploader("ad_dataset.xlsx", type=["xlsx", "xls"])
    if use_demo and not mv_file and not ad_file:
        mv_df = pd.DataFrame({"title":["Ocean Quest","Laugh Lane","Spy Night"],"genre":["Kids","Comedy","Thriller"],"duration_sec":[5400,6000,7200],"target_audience":["Kids","Teen","Adults"]})
        ad_df = pd.DataFrame({"ad_title":["ToyBlitz","ColaMax","BankPro"],"category":["Toys","Beverage","Finance"],"duration_sec":[30,45,30],"target":["Kids","Teen","Adults"],"importance":["priority","mid","priority"],"at_least_times_played":[2,0,1],"revenue_on_ad":[150.0,120.0,200.0]})
    else:
        mv_df = pd.read_excel(mv_file) if mv_file else pd.DataFrame()
        ad_df = pd.read_excel(ad_file) if ad_file else pd.DataFrame()
    movies = rows_to_movies(mv_df) if not mv_df.empty else []
    ads = rows_to_ads(ad_df) if not ad_df.empty else []
    st.subheader("Selections")
    mv_labels = [f"{m.title} | {m.target_audience}" for m in movies]
    ad_labels = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in ads]
    sel_movies = st.multiselect("Movies", mv_labels, default=mv_labels)
    sel_ads = st.multiselect("Ads", ad_labels, default=ad_labels)
    sel_movie_objs = [m for m in movies if f"{m.title} | {m.target_audience}" in sel_movies]
    sel_ad_objs = [a for a in ads if f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" in sel_ads]
    run = st.button("Run Schedule", type="primary")

col1, col2 = st.columns([2,1])
if run:
    base, _ = schedule_all_movies(sel_movie_objs)
    filled = fill_ad_breaks(base, sel_ad_objs)
    df = schedule_to_df(filled)
    with col1:
        st.subheader("Optimized Schedule")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "tv_schedule.csv", "text/csv")
        st.caption("Window 08:00–23:00. 3 parts/movie, 360s mid-rolls, 900s between movies. Ads: audience score → importance (until quota) → revenue/sec; avoid same-category back-to-back & duplicate titles in a break; shift times if a break underruns.")
    with col2:
        st.subheader("Ad Analysis")
        if sel_ad_objs:
            choice = st.selectbox("Select an ad", [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in sel_ad_objs])
            chosen = next(a for a in sel_ad_objs if f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" == choice)
            s = analyze_ad(filled, chosen)
            st.text_area("Details", value=(f"Ad Analysis: {chosen.ad_title}\n"
                f"• Required plays: {s['required']}\n"
                f"• Actual total plays: {s['total_plays']}\n"
                f"• % of quota fulfilled: {s['pct_quota']}%\n"
                f"• Plays on matching breaks: {s['match_plays']}\n"
