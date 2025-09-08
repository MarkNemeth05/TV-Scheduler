from __future__ import annotations
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# ─────────── Types ───────────
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

# ───────── Tables (same logic as before) ─────────
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

# ───────── Helpers ─────────
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

# ───────── Core logic ─────────
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
        used_time = int((cur_t - slot.start_time).total_seconds())
        slot_dur = int((slot.end_time - slot.start_time).total_seconds())
        shift = used_time - slot_dur
        if shift != 0:
            for j in range(i+1, len(sched)):
                sched[j].start_time += timedelta(seconds=shift)
                sched[j].end_time += timedelta(seconds=shift)
        i += 1
    return final

# ───────── IO helpers ─────────
def rows_to_movies(df: pd.DataFrame) -> List[Movie]:
    if df is None or df.empty: return []
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
    if df is None or df.empty: return []
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
#--------Summary helper-------------
def _fmt(t): 
    return t.strftime("%H:%M:%S")

def build_blocks(slots):
    """
    Turns the flat schedule into alternating 'movie part' and 'ad break' blocks.
    Each movie part is followed by a single break block whose revenue is the
    sum of all ad revenues until the next movie begins.
    """
    blocks = []
    i, n = 0, len(slots)
    while i < n:
        s = slots[i]
        if s.content_type == "movie":
            # movie card
            blocks.append({
                "kind": "movie",
                "title": s.title,
                "start": s.start_time,
                "end": s.end_time,
                "revenue": 0.0,
            })
            i += 1

            # collect the following ads (one break)
            rev, start, end = 0.0, None, None
            while i < n and slots[i].content_type == "ad":
                if start is None: start = slots[i].start_time
                rev += float(slots[i].revenue or 0.0)
                end = slots[i].end_time
                i += 1

            if start is not None and end is not None:
                blocks.append({
                    "kind": "break",
                    "title": "Ad Break",
                    "start": start,
                    "end": end,
                    "revenue": rev,
                })
        else:
            # (safety) skip anything unexpected
            i += 1
    return blocks

# ───────── UI (centered row of controls; results underneath) ─────────
st.set_page_config(page_title="TV Scheduler (ChatGPT × Márk Németh)", layout="wide")
st.markdown("<h1 style='text-align:center;margin-top:0;'>TV Scheduler</h1>", unsafe_allow_html=True)

# Initialize session state once
defaults = {
    "movies_df": None,
    "ads_df": None,
    "movies": [],
    "ads": [],
    "sel_movies": [],
    "sel_ads": [],
    "slots": [],
    "df": pd.DataFrame(),
    "ran": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# A single centered row with ALL controls
outer_left, center, outer_right = st.columns([1, 6, 1])
with center:
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.6, 1.6, 0.9])

    with c1:
        mv_file = st.file_uploader("movie_dataset.xlsx", type=["xlsx", "xls"], key="mv_upload")
    with c2:
        ad_file = st.file_uploader("ad_dataset.xlsx", type=["xlsx", "xls"], key="ad_upload")

    # Read uploads only when provided; otherwise keep previous data in state
    if mv_file is not None:
        st.session_state["movies_df"] = pd.read_excel(io.BytesIO(mv_file.getvalue()))
        st.session_state["movies"] = rows_to_movies(st.session_state["movies_df"])
        # refresh default selections
        st.session_state["sel_movies"] = [f"{m.title} | {m.target_audience}" for m in st.session_state["movies"]]

    if ad_file is not None:
        st.session_state["ads_df"] = pd.read_excel(io.BytesIO(ad_file.getvalue()))
        st.session_state["ads"] = rows_to_ads(st.session_state["ads_df"])
        st.session_state["sel_ads"] = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in st.session_state["ads"]]

    with c3:
        mv_labels = [f"{m.title} | {m.target_audience}" for m in st.session_state["movies"]]
        st.session_state["sel_movies"] = st.multiselect(
            "Movies", options=mv_labels, default=st.session_state["sel_movies"], key="mv_pick"
        )

    with c4:
        ad_labels = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in st.session_state["ads"]]
        st.session_state["sel_ads"] = st.multiselect(
            "Ads", options=ad_labels, default=st.session_state["sel_ads"], key="ad_pick"
        )

    def generate():
        movies = [m for m in st.session_state["movies"] if f"{m.title} | {m.target_audience}" in st.session_state["sel_movies"]]
        ads = [a for a in st.session_state["ads"] if f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" in st.session_state["sel_ads"]]
        base, _ = schedule_all_movies(movies)
        filled = fill_ad_breaks(base, ads)
        st.session_state["slots"] = filled
        st.session_state["df"] = schedule_to_df(filled)
        st.session_state["ran"] = True

    with c5:
        st.button("Generate", type="primary", use_container_width=True, on_click=generate)

# Results below the control row
with center:
    st.markdown("---")
    r1, r2 = st.columns([3, 2])

    with r1:
        st.subheader("Optimized Schedule")
        if st.session_state["ran"] and not st.session_state["df"].empty:
            st.dataframe(st.session_state["df"], use_container_width=True, height=420)
            csv = st.session_state["df"].to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="tv_schedule.csv", mime="text/csv")
            st.caption("Window: 08:00–23:00. Movies split into 3 parts with 360s mid-rolls; 900s between movies. Ads chosen by audience score → importance (until quota) → revenue/sec; avoid same-category back-to-back & duplicate titles; shift times when a break underruns.")
            # ── Compact visual timeline (cards) ─────────────────────────────────────
            blocks = build_blocks(st.session_state["slots"])
            if blocks:
                st.markdown("#### Timeline overview")
                # 3 cards per row
                for row_start in range(0, len(blocks), 3):
                cols = st.columns(3)
                for k, b in enumerate(blocks[row_start:row_start+3]):
                    with cols[k]:
                        bg = "#eef6ff" if b["kind"] == "movie" else "#fff7ed"
                        label = "Movie part" if b["kind"] == "movie" else "Ad break"
                        title = b["title"]
                        subtitle = f"{_fmt(b['start'])}–{_fmt(b['end'])}"
                        revenue = f"${b['revenue']:.0f}"

                        st.markdown(
                            f"""
                <div style="
                    border-radius:16px;padding:14px;
                    background:{bg};border:1px solid rgba(0,0,0,.06);
                    box-shadow:0 1px 2px rgba(0,0,0,.04);
                ">
                    <div style="font-weight:600;margin-bottom:6px;font-size:16px;line-height:1.2;">{title}</div>
                    <div style="font-size:12px;color:#666;margin-bottom:10px;">{subtitle}</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:12px;">{label}</span>
                        <span style="background:#111;color:#fff;border-radius:999px;padding:4px 10px;font-size:12px;">{revenue}</span>
                    </div>
                </div>
                """,
                            unsafe_allow_html=True,
                        )

        else:
            st.info("Upload both files, pick items, then click **Generate**.")

    with r2:
        st.subheader("Ad Analysis")
        if st.session_state["ran"] and st.session_state["slots"] and st.session_state["ads"]:
            ad_opts = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in st.session_state["ads"]]
            ad_choice = st.selectbox("Select an ad", options=ad_opts, key="ad_choice_select")
            chosen = next(a for a in st.session_state["ads"] if f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" == ad_choice)
            stats = analyze_ad(st.session_state["slots"], chosen)
            details = "\n".join([
                f"Ad Analysis: {chosen.ad_title}",
                f"• Required plays: {stats['required']}",
                f"• Actual total plays: {stats['total_plays']}",
                f"• % of quota fulfilled: {stats['pct_quota']}%",
                f"• Plays on matching breaks: {stats['match_plays']}",
                f"• % matching context: {stats['pct_match']}%",
            ])
            st.text_area("Details", value=details, height=240)
        else:
            st.info("Generate a schedule to analyze ad delivery.")
