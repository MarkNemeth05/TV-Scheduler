from __future__ import annotations
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# ───────────────────────── Types ─────────────────────────
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

# ─────────────────── Parameter Tables ───────────────────
TIME_TABLE: Dict[int, Dict[Audience, float]] = {}
for h in range(8, 12): TIME_TABLE[h] = {"Kids": 1.0, "Teen": 0.9, "Adults": 0.8}
for h in range(12, 16): TIME_TABLE[h] = {"Kids": 0.9, "Teen": 1.0, "Adults": 0.8}
for h in range(16, 18): TIME_TABLE[h] = {"Kids": 0.8, "Teen": 1.0, "Adults": 0.9}
for h in range(18, 23): TIME_TABLE[h] = {"Kids": 0.8, "Teen": 0.9, "Adults": 1.0}

GENRE_TABLE: Dict[str, Dict[Audience, float]] = {
    "Action":  {"Kids": 0.08, "Teen": 0.10, "Adults": 0.10},
    "Comedy":  {"Kids": 0.08, "Teen": 0.10, "Adults": 0.10},
    "Thriller":{"Kids": 0.08, "Teen": 0.09, "Adults": 0.10},
    "Kids":    {"Kids": 0.10, "Teen": 0.09, "Adults": 0.08},
}
MID_ROLL_TABLE: Dict[Audience, Dict[Audience, float]] = {
    "Kids": {"Kids": 0.10, "Teen": 0.09, "Adults": 0.08},
    "Teen": {"Kids": 0.09, "Teen": 0.10, "Adults": 0.08},
    "Adults": {"Kids": 0.08, "Teen": 0.09, "Adults": 0.10},
}
BETWEEN_TABLE: Dict[str, Dict[Audience, float]] = {
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

# ─────────────────────── Helpers ───────────────────────
def split_into_parts(duration: int) -> List[int]:
    base = duration // 3
    parts = [base, base, base]
    for i in range(duration - sum(parts)):
        parts[i % 3] += 1
    return parts

def today_at(h: int, m: int = 0, s: int = 0) -> datetime:
    now = datetime.now()
    return now.replace(hour=h, minute=m, second=s, microsecond=0)

def rps(ad: Ad) -> float:
    return ad.revenue_on_ad / max(1, ad.duration_sec)

# ────────────────── Core Scheduling ──────────────────
def schedule_all_movies(movies: List[Movie]) -> Tuple[List[Slot], List[Movie]]:
    schedule: List[Slot] = []
    pool = movies.copy()
    remaining = pool.copy()
    current = today_at(8, 0, 0)
    end_day = today_at(23, 0, 0)

    while pool and current < end_day:
        scored = []
        for m in pool:
            wt = TIME_TABLE.get(current.hour, {}).get(m.target_audience, 0.0)
            wg = GENRE_TABLE.get(m.genre, {}).get(m.target_audience, 0.0)
            scored.append((round(wt + wg, 4), m))
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
    final: List[Slot] = []
    sched = [Slot(**s.__dict__) for s in schedule_in]

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
        cur_t = slot.start_time
        last_cat: str | None = None
        used_titles = set()

        while True:
            cands = []
            for ad in ads:
                if ad.duration_sec > left: continue
                if ad.category == last_cat: continue
                if ad.ad_title in used_titles: continue

                played = usage.get(ad.ad_title, 0)
                eff_imp = "mid" if (ad.importance == "priority" and played >= ad.at_least_times_played) else ad.importance
                imp_score = 1.0 if eff_imp == "priority" else 0.9
                table = MID_ROLL_TABLE[prev_aud] if slot.title == "MID-ROLL" else BETWEEN_TABLE[f"{prev_aud}|{next_aud}"]
                base = table.get(ad.target, 0.0)
                cands.append((base, imp_score, rps(ad), ad, eff_imp))

            if not cands:
                break

            cands.sort(key=lambda x: (-x[0], -x[1], -x[2]))
            _, _, _, pick, eff_imp = cands[0]
            end_t = cur_t + timedelta(seconds=pick.duration_sec)
            final.append(Slot(cur_t, end_t, "ad", pick.ad_title, pick.target, eff_imp, pick.revenue_on_ad))
            used_titles.add(pick.ad_title)
            last_cat = pick.category
            cur_t = end_t
            left -= pick.duration_sec
            if eff_imp == "priority":
                usage[pick.ad_title] = usage.get(pick.ad_title, 0) + 1

        # shift downstream items
        used_time = int((cur_t - slot.start_time).total_seconds())
        slot_dur = int((slot.end_time - slot.start_time).total_seconds())
        shift = used_time - slot_dur
        if shift != 0:
            for j in range(i+1, len(sched)):
                sched[j].start_time += timedelta(seconds=shift)
                sched[j].end_time += timedelta(seconds=shift)
        i += 1

    return final

# ───────────────────── IO Helpers ─────────────────────
def rows_to_movies(df: pd.DataFrame) -> List[Movie]:
    if df.empty: return []
    def g(r, *names):
        for n in names:
            if n in df.columns: return r[n]
        return None
    movies: List[Movie] = []
    for _, r in df.iterrows():
        title = g(r, "title", "TITLE")
        genre = g(r, "genre", "main genre")
        dur = g(r, "duration_sec", "duration (sec)")
        ta = g(r, "target_audience")
        if pd.isna(title) or pd.isna(genre) or pd.isna(dur) or pd.isna(ta): continue
        movies.append(Movie(str(title), str(genre), int(dur), str(ta)))
    return movies

def rows_to_ads(df: pd.DataFrame) -> List[Ad]:
    if df.empty: return []
    def g(r, *names):
        for n in names:
            if n in df.columns: return r[n]
        return None
    ads: List[Ad] = []
    for _, r in df.iterrows():
        ad_title = g(r, "ad_title")
        category = g(r, "category")
        dur = g(r, "duration_sec", "duration (sec)")
        target = g(r, "target")
        importance = g(r, "importance") or "mid"
        minplays = g(r, "at_least_times_played") or 0
        revenue = g(r, "revenue_on_ad") or 0
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
        ctx = {prev.audience} if (prev and s.title == "MID-ROLL") else {prev.audience if prev else None, nextm.audience if nextm else None}
        if ad.target in ctx: match += 1
    return {"total_plays": total, "required": req, "pct_quota": round(quota, 1), "match_plays": match, "pct_match": round((match/total*100) if total else 0.0, 1)}

# ───────────────────────── UI ─────────────────────────
st.set_page_config(page_title="TV Scheduler (ChatGPT × Márk Németh)", layout="wide")

# Make everything visible and centered (no sidebar)
left, mid, right = st.columns([1, 5, 1])
with mid:
    st.title("TV Scheduler")
    st.caption("Made by ChatGPT, based on the work by Márk Németh")

    # Session defaults
    for k, v in {
        "ran": False,
        "slots": [],
        "df": pd.DataFrame(),
        "movies_list": [],
        "ads_list": [],
    }.items():
        if k not in st.session_state: st.session_state[k] = v

    st.subheader("Load data")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        mv_file = st.file_uploader("movie_dataset.xlsx", type=["xlsx", "xls"], key="mv_upl")
    with c2:
        ad_file = st.file_uploader("ad_dataset.xlsx", type=["xlsx", "xls"], key="ad_upl")
    with c3:
        use_demo = st.checkbox("Use demo data", value=False)

    # Build dataframes
    if use_demo and (mv_file is None and ad_file is None):
        mv_df = pd.DataFrame({
            "title": ["Ocean Quest", "Laugh Lane", "Spy Night"],
            "genre": ["Kids", "Comedy", "Thriller"],
            "duration_sec": [5400, 6000, 7200],
            "target_audience": ["Kids", "Teen", "Adults"],
        })
        ad_df = pd.DataFrame({
            "ad_title": ["ToyBlitz", "ColaMax", "BankPro"],
            "category": ["Toys", "Beverage", "Finance"],
            "duration_sec": [30, 45, 30],
            "target": ["Kids", "Teen", "Adults"],
            "importance": ["priority", "mid", "priority"],
            "at_least_times_played": [2, 0, 1],
            "revenue_on_ad": [150.0, 120.0, 200.0],
        })
    else:
        mv_df = pd.read_excel(mv_file) if mv_file else pd.DataFrame()
        ad_df = pd.read_excel(ad_file) if ad_file else pd.DataFrame()

    movies = rows_to_movies(mv_df) if not mv_df.empty else []
    ads = rows_to_ads(ad_df) if not ad_df.empty else []
    st.session_state["movies_list"] = movies
    st.session_state["ads_list"] = ads

    st.subheader("Selections")
    mv_labels = [f"{m.title} | {m.target_audience}" for m in movies]
    ad_labels = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in ads]

    # Always-visible selectors in the middle
    s1, s2 = st.columns(2)
    with s1:
        sel_movies = st.multiselect("Movies", options=mv_labels, default=mv_labels, key="mv_sel")
    with s2:
        sel_ads = st.multiselect("Ads", options=ad_labels, default=ad_labels, key="ad_sel")

    sel_movie_objs = [m for m in movies if f"{m.title} | {m.target_audience}" in sel_movies]
    sel_ad_objs = [a for a in ads if f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" in sel_ads]

    # Generate button that PERSISTS state across reruns
    def generate():
        base, _ = schedule_all_movies(sel_movie_objs)
        filled = fill_ad_breaks(base, sel_ad_objs)
        st.session_state["slots"] = filled
        st.session_state["df"] = schedule_to_df(filled)
        st.session_state["ran"] = True

    st.button("Generate Schedule", type="primary", on_click=generate)

    # Results + Analysis stay visible after any interaction
    r1, r2 = st.columns([3, 2])
    with r1:
        st.subheader("Optimized Schedule")
        if st.session_state["ran"] and not st.session_state["df"].empty:
            st.dataframe(st.session_state["df"], use_container_width=True)
            csv = st.session_state["df"].to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="tv_schedule.csv", mime="text/csv")
            st.caption("Window: 08:00–23:00. Movies split into 3 parts with 360s mid-rolls; 900s between movies. Ads: audience score → importance (until quota) → revenue/sec; avoid same-category back-to-back & duplicate titles in a break; shift times if a break underruns.")
        else:
            st.info("Load data, choose selections, then click **Generate Schedule**.")

    with r2:
        st.subheader("Ad Analysis")
        if st.session_state["ran"] and st.session_state["slots"]:
            ad_opts = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in st.session_state["ads_list"]]
            if ad_opts:
                ad_choice = st.selectbox("Select an ad", options=ad_opts, key="ad_choice")
                chosen = next(a for a in st.session_state["ads_list"] if f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" == ad_choice)
                stats = analyze_ad(st.session_state["slots"], chosen)
                details = "\n".join([
                    f"Ad Analysis: {chosen.ad_title}",
                    f"• Required plays: {stats['required']}",
                    f"• Actual total plays: {stats['total_plays']}",
                    f"• % of quota fulfilled: {stats['pct_quota']}%",
                    f"• Plays on matching breaks: {stats['match_plays']}",
                    f"• % matching context: {stats['pct_match']}%",
                ])
                st.text_area("Details", value=details, height=200)
            else:
                st.info("No ads loaded yet.")
        else:
            st.info("Generate a schedule to analyze ad delivery.")
