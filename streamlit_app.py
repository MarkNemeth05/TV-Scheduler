# streamlit_app.py — TV Scheduler (ChatGPT build, based on work by Márk Németh)
from __future__ import annotations
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# ===================== Types =====================
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

# ===================== Parameter Tables =====================
TIME_TABLE: Dict[int, Dict[Audience, float]] = {}
for h in range(8, 12):
    TIME_TABLE[h] = {"Kids": 1.0, "Teen": 0.9, "Adults": 0.8}
for h in range(12, 16):
    TIME_TABLE[h] = {"Kids": 0.9, "Teen": 1.0, "Adults": 0.8}
for h in range(16, 18):
    TIME_TABLE[h] = {"Kids": 0.8, "Teen": 1.0, "Adults": 0.9}
for h in range(18, 23):
    TIME_TABLE[h] = {"Kids": 0.8, "Teen": 0.9, "Adults": 1.0}

GENRE_TABLE: Dict[str, Dict[Audience, float]] = {
    "Action":  {"Kids": 0.08, "Teen": 0.10, "Adults": 0.10},
    "Comedy":  {"Kids": 0.08, "Teen": 0.10, "Adults": 0.10},
    "Thriller":{"Kids": 0.08, "Teen": 0.09, "Adults": 0.10},
    "Kids":    {"Kids": 0.10, "Teen": 0.09, "Adults": 0.08},
}

MID_ROLL_TABLE: Dict[Audience, Dict[Audience, float]] = {
    "Kids":   {"Kids": 0.10, "Teen": 0.09, "Adults": 0.08},
    "Teen":   {"Kids": 0.09, "Teen": 0.10, "Adults": 0.08},
    "Adults": {"Kids": 0.08, "Teen": 0.09, "Adults": 0.10},
}

BETWEEN_TABLE: Dict[str, Dict[Audience, float]] = {
    "Kids|Kids":   {"Kids": 0.10, "Teen": 0.09, "Adults": 0.08},
    "Kids|Teen":   {"Kids": 0.10, "Teen": 0.10, "Adults": 0.09},
    "Kids|Adults": {"Kids": 0.10, "Teen": 0.10, "Adults": 0.10},
    "Teen|Kids":   {"Kids": 0.10, "Teen": 0.10, "Adults": 0.09},
    "Teen|Teen":   {"Kids": 0.09, "Teen": 0.10, "Adults": 0.09},
    "Teen|Adults": {"Kids": 0.09, "Teen": 0.10, "Adults": 0.10},
    "Adults|Kids": {"Kids": 0.10, "Teen": 0.10, "Adults": 0.10},
    "Adults|Teen": {"Kids": 0.09, "Teen": 0.10, "Adults": 0.10},
    "Adults|Adults": {"Kids": 0.08, "Teen": 0.09, "Adults": 0.10},
}

# ===================== Helpers =====================
def split_into_parts(duration: int) -> List[int]:
    base = duration // 3
    parts = [base, base, base]
    rem = duration - sum(parts)
    for i in range(rem):
        parts[i % 3] += 1
    return parts

def today_at(h: int, m: int = 0, s: int = 0) -> datetime:
    now = datetime.now()
    return now.replace(hour=h, minute=m, second=s, microsecond=0)

def rps(ad: Ad) -> float:
    return ad.revenue_on_ad / max(1, ad.duration_sec)

# ===================== Core Scheduling =====================
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
        parts = split_into_parts(best.duration_sec)

        for i, dur in enumerate(parts):
            end = current + timedelta(seconds=dur)
            schedule.append(Slot(
                start_time=current,
                end_time=end,
                content_type="movie",
                title=f"{best.title} (Part {i+1}/3)",
                audience=best.target_audience,
                revenue=0.0,
            ))
            current = end
            if i < 2:  # mid-roll break
                br_end = current + timedelta(seconds=360)
                schedule.append(Slot(
                    start_time=current,
                    end_time=br_end,
                    content_type="ad_break",
                    title="MID-ROLL",
                    audience=best.target_audience,
                ))
                current = br_end

        # remove chosen movie
        pool.remove(best)
        remaining = [m for m in remaining if m.title != best.title]

        if pool and current < end_day:
            br_end = current + timedelta(seconds=900)
            schedule.append(Slot(
                start_time=current,
                end_time=br_end,
                content_type="ad_break",
                title="BETWEEN-MOVIE",
                audience=best.target_audience,
            ))
            current = br_end

    return schedule, remaining

def fill_ad_breaks(schedule_in: List[Slot], ads: List[Ad]) -> List[Slot]:
    usage: Dict[str, int] = {}  # priority plays used per ad title
    final: List[Slot] = []
    sched = [Slot(**s.__dict__) for s in schedule_in]  # shallow copy

    i = 0
    while i < len(sched):
        slot = sched[i]
        if slot.content_type != "ad_break":
            final.append(slot)
            i += 1
            continue

        # Prev/next movie audiences
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
                if ad.duration_sec > left:
                    continue
                if ad.category == last_cat:
                    continue
                if ad.ad_title in used_titles:
                    continue

                played = usage.get(ad.ad_title, 0)
                eff_imp = ad.importance
                if eff_imp == "priority" and played >= ad.at_least_times_played:
                    eff_imp = "mid"
                imp_score = 1.0 if eff_imp == "priority" else 0.9

                table = MID_ROLL_TABLE[prev_aud] if slot.title == "MID-ROLL" else BETWEEN_TABLE[f"{prev_aud}|{next_aud}"]
                base = table.get(ad.target, 0.0)
                cand = (base, imp_score, rps(ad), ad, eff_imp)
                cands.append(cand)

            if not cands:
                break

            cands.sort(key=lambda x: (-x[0], -x[1], -x[2]))
            _, _, _, ad_pick, eff_imp = cands[0]

            end_t = cur_t + timedelta(seconds=ad_pick.duration_sec)
            final.append(Slot(
                start_time=cur_t,
                end_time=end_t,
                content_type="ad",
                title=ad_pick.ad_title,
                audience=ad_pick.target,
                importance=eff_imp,
                revenue=ad_pick.revenue_on_ad,
            ))
            used_titles.add(ad_pick.ad_title)
            last_cat = ad_pick.category
            cur_t = end_t
            left -= ad_pick.duration_sec
            if eff_imp == "priority":
                usage[ad_pick.ad_title] = usage.get(ad_pick.ad_title, 0) + 1

        # shift downstream items by (used_time - slot_duration)
        used_time = int((cur_t - slot.start_time).total_seconds())
        slot_duration = int((slot.end_time - slot.start_time).total_seconds())
        shift_amt = used_time - slot_duration  # can be negative (underrun)
        if shift_amt != 0:
            for j in range(i+1, len(sched)):
                sched[j].start_time = sched[j].start_time + timedelta(seconds=shift_amt)
                sched[j].end_time = sched[j].end_time + timedelta(seconds=shift_amt)
        i += 1

    return final

# ===================== IO Helpers =====================
def rows_to_movies(df: pd.DataFrame) -> List[Movie]:
    # Accept either snake_case or original headers
    cols = {c.lower(): c for c in df.columns}
    def get(row, key, fallback=None):
        if key in cols:
            return row[cols[key]]
        if fallback and fallback in cols:
            return row[cols[fallback]]
        return None

    movies: List[Movie] = []
    for _, r in df.iterrows():
        title = get(r, "title", "title") or get(r, "title", "TITLE")
        genre = get(r, "genre", "main genre")
        dur = get(r, "duration_sec", "duration (sec)")
        ta = get(r, "target_audience", "target_audience")
        if pd.isna(title) or pd.isna(genre) or pd.isna(dur) or pd.isna(ta):
            continue
        try:
            movies.append(Movie(str(title), str(genre), int(dur), str(ta)))
        except Exception:
            pass
    return movies

def rows_to_ads(df: pd.DataFrame) -> List[Ad]:
    cols = {c.lower(): c for c in df.columns}
    def get(row, key, fallback=None):
        if key in cols:
            return row[cols[key]]
        if fallback and fallback in cols:
            return row[cols[fallback]]
        return None

    ads: List[Ad] = []
    for _, r in df.iterrows():
        ad_title = get(r, "ad_title")
        category = get(r, "category")
        dur = get(r, "duration_sec", "duration (sec)")
        target = get(r, "target")
        importance = get(r, "importance") or "mid"
        minplays = get(r, "at_least_times_played") or 0
        revenue = get(r, "revenue_on_ad") or 0
        if pd.isna(ad_title) or pd.isna(category) or pd.isna(dur) or pd.isna(target):
            continue
        try:
            ads.append(Ad(
                ad_title=str(ad_title),
                category=str(category),
                duration_sec=int(dur),
                target=str(target),
                importance=str(importance),
                at_least_times_played=int(minplays),
                revenue_on_ad=float(revenue),
            ))
        except Exception:
            pass
    return ads

def schedule_to_df(slots: List[Slot]) -> pd.DataFrame:
    rows = []
    for s in slots:
        rev_per_sec = (
            s.revenue / max(1, int((s.end_time - s.start_time).total_seconds()))
            if s.content_type == "ad" else None
        )
        rows.append({
            "start": s.start_time.strftime("%H:%M:%S"),
            "end": s.end_time.strftime("%H:%M:%S"),
            "title": s.title,
            "type": s.content_type,
            "audience": s.audience,
            "importance": s.importance or "-",
            "rev_per_sec": None if rev_per_sec is None else round(rev_per_sec, 2),
        })
    return pd.DataFrame(rows)

def analyze_ad(slots: List[Slot], ad: Ad) -> Dict[str, float | int]:
    total_plays = sum(1 for s in slots if s.content_type == "ad" and s.title == ad.ad_title)
    required = ad.at_least_times_played
    pct_quota = (total_plays / required * 100) if required > 0 else 0.0

    match_plays = 0
    for i, s in enumerate(slots):
        if s.content_type != "ad" or s.title != ad.ad_title:
            continue
        prev = next((x for x in reversed(slots[:i]) if x.content_type == "movie"), None)
        nextm = next((x for x in slots[i+1:] if x.content_type == "movie"), None)
        context_aud = {prev.audience} if prev and s.title == "MID-ROLL" else {
            prev.audience if prev else None,
            nextm.audience if nextm else None
        }
        if ad.target in context_aud:
            match_plays += 1
    pct_match = (match_plays / total_plays * 100) if total_plays > 0 else 0.0
    return {
        "total_plays": int(total_plays),
        "required": int(required),
        "pct_quota": round(pct_quota, 1),
        "match_plays": int(match_plays),
        "pct_match": round(pct_match, 1),
    }

# ===================== UI =====================
st.set_page_config(page_title="TV Scheduler (ChatGPT × Márk Németh)", layout="wide")
st.title("TV Scheduler")
st.caption("Made by ChatGPT, based on the work by Márk Németh")

with st.sidebar:
    st.header("Load data")
    use_demo = st.toggle("Use small demo data", value=False, help="Populate a tiny sample if you don't have files yet.")
    mv_file = st.file_uploader("movie_dataset.xlsx", type=["xlsx", "xls"])
    ad_file = st.file_uploader("ad_dataset.xlsx", type=["xlsx", "xls"])

    if use_demo and not mv_file and not ad_file:
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

    st.subheader("Selections")
    mv_labels = [f"{m.title} | {m.target_audience}" for m in movies]
    ad_labels = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in ads]
    sel_movies = st.multiselect("Movies", options=mv_labels, default=mv_labels)
    sel_ads = st.multiselect("Ads", options=ad_labels, default=ad_labels)

    sel_movie_objs = [m for m in movies if f"{m.title} | {m.target_audience}" in sel_movies]
    sel_ad_objs = [a for a in ads if f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" in sel_ads]

    run = st.button("Run Schedule", type="primary")

col1, col2 = st.columns([2, 1])

if run:
    base, _ = schedule_all_movies(sel_movie_objs)
    filled = fill_ad_breaks(base, sel_ad_objs)

    df = schedule_to_df(filled)

    with col1:
        st.subheader("Optimized Schedule")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="tv_schedule.csv", mime="text/csv")
        st.caption("Window: 08:00–23:00. Movies split into 3 parts with 360s mid-rolls. 900s between movies. Ads chosen by audience score → importance (until quota met) → revenue/sec, avoiding back-to-back same categories and duplicate titles within a break. Downstream slots shift if a break underruns.")

    with col2:
        st.subheader("Ad Analysis")
        if sel_ad_objs:
            ad_choice = st.selectbox(
                "Select an ad",
                options=[f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in sel_ad_objs],
            )
            chosen = next(
                a for a in sel_ad_objs
                if f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" == ad_choice
            )
            stats = analyze_ad(filled, chosen)
            st.text_area(
                "Details",
                value=(
                    f"Ad Analysis: {chosen.ad_title}\n"
                    f"• Required plays: {stats['required']}\n"
                    f"• Actual total plays: {stats['total_plays']}\n"
                    f"• % of quota fulfilled: {stats['pct_quota']}%\n"
                    f"• Plays on matching breaks: {stats['match_plays']}\n"
                    f"• % matching context: {stats['pct_match']}%\n"
                ),
                height=180,
            )
        else:
            st.info("Load ads to analyze their delivery and context match.")
else:
    st.info("Upload your spreadsheets (or toggle demo data), pick items in the sidebar, then click **Run Schedule**.")
