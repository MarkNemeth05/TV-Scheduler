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

# ───────── Tables (same scheduling logic) ─────────
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

# ───────── Summary builders ─────────
def build_blocks(slots: List[Slot]):
    """Alternate movie part rows and the immediately following ad-break (with total revenue)."""
    blocks = []
    i, n = 0, len(slots)
    while i < n:
        s = slots[i]
        if s.content_type == "movie":
            blocks.append({"kind":"movie","title":s.title,"start":s.start_time,"end":s.end_time,"revenue":0.0})
            i += 1
            # ads until next movie = one break
            rev, start, end = 0.0, None, None
            while i < n and slots[i].content_type == "ad":
                if start is None: start = slots[i].start_time
                rev += float(slots[i].revenue or 0.0)
                end = slots[i].end_time
                i += 1
            if start is not None and end is not None:
                blocks.append({"kind":"break","title":"Ad Break","start":start,"end":end,"revenue":rev})
        else:
            i += 1
    return blocks

def hourly_revenue_cumulative(slots: List[Slot]) -> pd.DataFrame:
    """Revenue per hour bucket: 08–09, 09–10, … , 22–23 (not cumulative)."""
    start = today_at(8, 0, 0)
    end = today_at(23, 0, 0)

    # Hour bucket starts: 08:00..22:00 (each represents [hh:00, hh+1:00))
    bucket_index = pd.date_range(start=start, end=end, freq="H")[:-1]

    if not slots:
        return pd.DataFrame({"revenue": [0.0] * len(bucket_index)}, index=bucket_index)

    rows = [
        {"time": s.end_time, "rev": float(s.revenue or 0.0)}
        for s in slots
        if s.content_type == "ad"
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"revenue": [0.0] * len(bucket_index)}, index=bucket_index)

    # Sum revenue into hourly buckets labeled by the left edge (hh:00)
    hourly = (
        df.set_index("time")["rev"]
        .sort_index()
        .resample("H", label="left", closed="left")
        .sum()
        .reindex(bucket_index, fill_value=0.0)
    )

    return pd.DataFrame({"revenue": hourly})


# ───────── UI (centered controls; summary table + chart) ─────────
st.set_page_config(page_title="TV Scheduler (ChatGPT × Márk Németh)", layout="wide")
st.markdown("<h1 style='text-align:center;margin-top:0;'>TV Scheduler</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Made by ChatGPT, based on the work by Márk Németh</p>", unsafe_allow_html=True)

# CSS: sticky table look, uses the same red tone as Streamlit tags for movie rows
st.markdown("""
<style>
.summary-table { width:100%; border-collapse: collapse; table-layout: fixed; }
.summary-table th, .summary-table td {
  padding: 10px 12px; border: 1px solid rgba(255,255,255,0.08);
  font-size: 14px; line-height: 1.2;
}
.summary-table th { position: sticky; top: 0; background: rgba(0,0,0,0.25); backdrop-filter: blur(4px); z-index: 2; }
.row-movie { background: #ef4444; color: white; }  /* red-500 like the multiselect chips */
.row-break { background: rgba(255,255,255,0.04); color: inherit; }
.badge {
  background: #111; color: #fff; border-radius: 999px; padding: 4px 10px; font-size: 12px;
  display: inline-block;
}
.timecell { font-weight: 700; width: 28%; font-variant-numeric: tabular-nums; }
.titlecell { width: 52%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.revcell  { width: 20%; text-align: right; }
.row-movie .titlecell { font-weight: 700; letter-spacing: .2px; }
.row-break .titlecell { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Initialize session state once
defaults = {"movies_df": None, "ads_df": None, "movies": [], "ads": [], "sel_movies": [], "sel_ads": [], "slots": [], "df": pd.DataFrame(), "ran": False}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# Controls row
outer_left, center, outer_right = st.columns([1, 6, 1])
with center:
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.6, 1.6, 0.9])

    with c1:
        mv_file = st.file_uploader("movie_dataset.xlsx", type=["xlsx", "xls"], key="mv_upload")
    with c2:
        ad_file = st.file_uploader("ad_dataset.xlsx", type=["xlsx", "xls"], key="ad_upload")

    if mv_file is not None:
        st.session_state["movies_df"] = pd.read_excel(io.BytesIO(mv_file.getvalue()))
        st.session_state["movies"] = rows_to_movies(st.session_state["movies_df"])
        st.session_state["sel_movies"] = [f"{m.title} | {m.target_audience}" for m in st.session_state["movies"]]

    if ad_file is not None:
        st.session_state["ads_df"] = pd.read_excel(io.BytesIO(ad_file.getvalue()))
        st.session_state["ads"] = rows_to_ads(st.session_state["ads_df"])
        st.session_state["sel_ads"] = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in st.session_state["ads"]]

    with c3:
        mv_labels = [f"{m.title} | {m.target_audience}" for m in st.session_state["movies"]]
        st.session_state["sel_movies"] = st.multiselect("Movies", options=mv_labels, default=st.session_state["sel_movies"], key="mv_pick")

    with c4:
        ad_labels = [f"{a.ad_title} | {a.target} | {a.importance} | {rps(a):.2f}" for a in st.session_state["ads"]]
        st.session_state["sel_ads"] = st.multiselect("Ads", options=ad_labels, default=st.session_state["sel_ads"], key="ad_pick")

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

# Results
with center:
    st.markdown("---")
    left_area, right_area = st.columns([3, 2])

    with left_area:
        st.subheader("Summary")
        if st.session_state["ran"] and st.session_state["slots"]:
            blocks = build_blocks(st.session_state["slots"])

            # Render compact, stuck table
            html = ["<div style='max-height:520px;overflow:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.08);'>"]
            html.append("<table class='summary-table'>")
            html.append("<thead><tr><th class='timecell'>Time</th><th class='titlecell'>Title</th><th class='revcell'>Revenue</th></tr></thead><tbody>")
            for b in blocks:
                row_cls = "row-movie" if b["kind"] == "movie" else "row-break"
                time_str = f"{b['start'].strftime('%H:%M')} – {b['end'].strftime('%H:%M')}"
                title = b["title"]
                rev = f"${b['revenue']:.0f}"
                html.append(
                    f"<tr class='{row_cls}'>"
                    f"<td class='timecell'>{time_str}</td>"
                    f"<td class='titlecell'>{title}</td>"
                    f"<td class='revcell'><span class='badge'>{rev}</span></td>"
                    f"</tr>"
                )
            html.append("</tbody></table></div>")
            st.markdown("\n".join(html), unsafe_allow_html=True)

            # Also keep the classic CSV/table download for power users
            with st.expander("Detailed table (export)"):
                st.dataframe(st.session_state["df"], use_container_width=True, height=320)
                csv = st.session_state["df"].to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="tv_schedule.csv", mime="text/csv")

        else:
            st.info("Upload both files, choose items, then click **Generate**.")

    with right_area:
        st.subheader("Revenue by Hour")
        if st.session_state["ran"] and st.session_state["slots"]:
            rev_df = hourly_revenue_cumulative(st.session_state["slots"])  # now returns per-hour
            # Labels like "08–09", "09–10", …
            rev_df.index = [
                f"{idx.strftime('%H')}-{(idx + pd.Timedelta(hours=1)).strftime('%H')}"
                for idx in rev_df.index
            ]
            st.line_chart(rev_df, height=360, use_container_width=True)
            total_rev = float(rev_df["revenue"].sum()) if not rev_df.empty else 0.0
            st.caption(f"Hourly revenue (08–23). Total: ${total_rev:,.0f}")

        else:
            st.info("Generate a schedule to see revenue over the day.")

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
            st.text_area("Details", value=details, height=160)
        else:
            st.info("Generate a schedule to analyze ad delivery.")
