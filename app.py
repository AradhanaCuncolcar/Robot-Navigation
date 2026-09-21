"""
app.py - Streamlit dashboard for the MDP wall-following robot.

    streamlit run app.py
"""
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import pipeline
import algorithms as A
import compare
import world
from data_loader import load_raw24, load_dataset, verify_24_to_4
from value_iteration import policy_df, value_df

st.set_page_config(page_title="Wall-following robot navigator", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- look & feel
INK, MUTED, LINE, PANEL = "#16222E", "#6B7785", "#D3DAE1", "#F6F8FA"
ACOL = {"Move-Forward": "#17906F", "Slight-Right-Turn": "#2F6FDB",
        "Sharp-Right-Turn": "#D2413A", "Slight-Left-Turn": "#7C4FE0"}
ANAME = {"Move-Forward": "Move forward", "Slight-Right-Turn": "Slight right turn",
         "Sharp-Right-Turn": "Sharp right turn", "Slight-Left-Turn": "Slight left turn"}
SCOL = {"SD_front": "#2F5D8A", "SD_left": "#17906F", "SD_right": "#C98A00", "SD_back": "#8A6FB0"}

ALGO_COLORS = ["#2F6FDB", "#E07A10", "#139A6B", "#B0379E"]
st.markdown(f"""
<style>
/* ---------- page frame ---------- */
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1480px; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: 0; }}
h4 {{ margin-top: .6rem; }}
.explain {{ color: #44525F; max-width: 80ch; }}

/* ---------- hero: mission-control header ---------- */
.hero {{ position: relative; overflow: hidden; background: #14212E; color: #fff; border-radius: 16px;
  padding: 26px 30px 24px; margin-bottom: 18px; display: grid; grid-template-columns: auto 1fr auto;
  gap: 26px; align-items: center; }}
.hero::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 5px;
  background: linear-gradient(90deg, #2F6FDB 0 25%, #E07A10 25% 50%, #139A6B 50% 75%, #B0379E 75% 100%); }}
.hero h1 {{ color: #fff; font-size: 2.3rem; line-height: 1.1; margin: 0; padding: 0; font-weight: 700; }}
.hero p {{ color: #B9C6D3; margin: 8px 0 12px; max-width: 72ch; font-size: 1.02rem; line-height: 1.5; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ display: inline-flex; align-items: center; gap: 7px; background: #1F3144; color: #E6EDF5;
  border-radius: 999px; padding: 4px 12px 4px 8px; font-size: .88rem; font-weight: 600; }}
.chip i {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.stats {{ display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 10px; }}
.stat {{ background: #1C2D3F; border-radius: 12px; padding: 12px 14px; border-top: 3px solid var(--c); }}
.stat b {{ display: block; font-size: 1.65rem; line-height: 1.1; font-variant-numeric: tabular-nums; color: #fff; }}
.stat span {{ color: #9FB0C1; font-size: .82rem; }}
@media (max-width: 1100px) {{ .hero {{ grid-template-columns: auto 1fr; }} .stats {{ grid-column: 1 / -1; }} }}
@media (max-width: 640px) {{ .hero {{ grid-template-columns: 1fr; }} .hero svg {{ display: none; }}
  .stats {{ grid-template-columns: 1fr 1fr; }} }}

/* ---------- tabs as a segmented control (old and new Streamlit markup) ---------- */
.stTabs [role="tablist"] {{ gap: 6px; background: #fff; border: 1px solid {LINE}; border-radius: 12px;
  padding: 5px; overflow-x: auto; box-shadow: none; }}
.stTabs [role="tab"], .stTabs [data-testid="stTab"] {{ height: auto; padding: 8px 16px; border-radius: 8px;
  font-weight: 600; color: #44525F; white-space: nowrap; border: 0; }}
.stTabs [role="tab"] p {{ font-weight: 600; }}
.stTabs [role="tab"]:hover {{ background: #EEF3FA; color: {INK}; }}
.stTabs [role="tab"][aria-selected="true"] {{ background: #14212E; color: #fff !important; }}
.stTabs [role="tab"][aria-selected="true"] p, .stTabs [role="tab"][aria-selected="true"] span {{ color: #fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [role="tabpanel"] {{ padding-top: 14px; }}

/* ---------- cards: metrics, charts, tables ---------- */
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-left: 5px solid #2F6FDB;
  border-radius: 12px; padding: 12px 16px; }}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {{ border-left-color: #E07A10; }}
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {{ border-left-color: #139A6B; }}
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {{ border-left-color: #B0379E; }}
[data-testid="stMetricValue"] {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
[data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px;
  padding: 6px 8px; }}
[data-testid="stDataFrame"] {{ background: #fff; border-radius: 10px; }}
[data-testid="stMain"] [data-testid="stExpander"] details {{ background: #fff; }}

/* ---------- mission result cards ---------- */
.results {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 6px 0 10px; }}
@media (max-width: 1100px) {{ .results {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 560px) {{ .results {{ grid-template-columns: 1fr; }} }}
.rc {{ background: #fff; border: 1px solid {LINE}; border-top: 5px solid var(--c); border-radius: 12px; padding: 14px 16px; }}
.rc .hd {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
.rc .hd b {{ font-size: 1.08rem; flex: 1; }}
.badge {{ font-size: .78rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; }}
.ok {{ background: #DFF5EA; color: #0E7A50; }}
.no {{ background: #FBE3E1; color: #B3261E; }}
.rc .big {{ display: flex; gap: 18px; margin-bottom: 8px; }}
.rc .big div b {{ display: block; font-size: 1.5rem; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.rc .big div span, .rc .row span {{ color: {MUTED}; font-size: .82rem; }}
.rc .row {{ display: flex; justify-content: space-between; border-top: 1px solid #EDF1F5; padding-top: 6px;
  margin-top: 6px; font-size: .9rem; font-variant-numeric: tabular-nums; }}
.meter {{ height: 6px; background: #EDF1F5; border-radius: 3px; overflow: hidden; margin-top: 4px; }}
.meter i {{ display: block; height: 100%; background: var(--c); }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] h3 {{ color: #fff; font-size: 1.05rem; margin-top: .8rem; }}
.side-brand {{ display: flex; align-items: center; gap: 10px; margin: 4px 0 6px; color: #fff; font-weight: 700;
  font-size: 1.05rem; }}
.side-note {{ color: #9FB0C1; font-size: .85rem; line-height: 1.45; }}
</style>""", unsafe_allow_html=True)


def robot_svg(size=64, body="#F2B705"):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" aria-hidden="true">'
            f'<line x1="32" y1="9" x2="36" y2="3" stroke="#C9D4DF" stroke-width="2.4" stroke-linecap="round"/>'
            f'<circle cx="36.5" cy="3.5" r="3" fill="#2F6FDB"/>'
            f'<circle cx="32" cy="36" r="26" fill="{body}"/>'
            f'<circle cx="32" cy="33" r="20" fill="#F4F7FA" stroke="#0E1822" stroke-width="2"/>'
            f'<rect x="16" y="23" width="32" height="19" rx="9" fill="#14212E"/>'
            f'<rect x="22" y="27.5" width="6" height="10" rx="3" fill="#8FF1FF"/>'
            f'<rect x="36" y="27.5" width="6" height="10" rx="3" fill="#8FF1FF"/>'
            f'<rect x="4" y="30" width="7" height="14" rx="3" fill="#0E1822"/>'
            f'<rect x="53" y="30" width="7" height="14" rx="3" fill="#0E1822"/></svg>')


def style_fig(fig, height=360, title=None, legend_cols=None, **kw):
    """White card styling. The title sits at the very top and the legend gets its own row
    underneath it, so the two can never overlap."""
    has_legend = sum(1 for tr in fig.data if tr.showlegend is not False) > 1 or kw.get("showlegend")
    n_leg = sum(1 for tr in fig.data if tr.showlegend is not False)
    rows = -(-n_leg // legend_cols) if legend_cols else 1
    top = 12 + (34 if title else 0) + ((34 + 22 * (rows - 1)) if has_legend else 0)
    fig.update_layout(
        height=height + top - 36, margin=dict(l=12, r=12, t=top, b=12),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Barlow, sans-serif", size=13, color=INK),
        title=dict(text=title, x=0.01, xanchor="left", y=1, yanchor="top", yref="container",
                   pad=dict(t=12), font=dict(family="Barlow Semi Condensed, sans-serif", size=17)) if title else None,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", yref="paper",
                    bgcolor="rgba(0,0,0,0)",
                    **({"entrywidth": 1 / legend_cols, "entrywidthmode": "fraction"} if legend_cols else {})),
        hoverlabel=dict(font_family="Barlow, sans-serif"), **kw)
    fig.update_xaxes(gridcolor="#EDF0F3", zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor="#EDF0F3", zerolinecolor=LINE, linecolor=LINE)
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f'<div class="side-brand">{robot_svg(34)}<span>Robot navigator</span></div>'
                '<div class="side-note">Settings retrain the policy instantly. Mission results for new '
                'settings take a minute or two to train.</div>', unsafe_allow_html=True)
    st.markdown("### Model settings")
    variant = st.radio("Sensor dataset", list(config.VARIANTS),
                       format_func=lambda k: config.VARIANTS[k]["label"], index=0)
    gamma = st.slider("Discount factor γ", 0.50, 0.99, config.GAMMA, 0.01,
                      help="How much future reward counts. Higher = more far-sighted.")
    tol = st.select_slider("Convergence threshold", options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                           value=config.TOLERANCE, format_func=lambda v: f"{v:.0e}",
                           help="Value iteration stops when no state value changes by more than this.")
    train_frac = st.slider("Training share of data", 0.5, 0.95, config.TRAIN_FRACTION, 0.05,
                           help="The first part of the time-ordered recording builds the MDP; "
                                "the rest tests the policy.")
    with st.expander("Reward design"):
        rw_ideal = st.slider("Left wall at ideal distance", 0.0, 5.0, config.REWARDS["left"]["ideal"], 0.5)
        rw_close = st.slider("Left wall too close", -5.0, 0.0, config.REWARDS["left"]["too_close"], 0.5)
        rw_far = st.slider("Left wall far", -5.0, 0.0, config.REWARDS["left"]["far"], 0.5)
        rw_lost = st.slider("Left wall lost", -5.0, 0.0, config.REWARDS["left"]["lost"], 0.5)
        rw_front = st.slider("Obstacle close in front", -6.0, 0.0, config.REWARDS["front"]["near"], 0.5)
        rw_imit = st.slider("Bonus for matching the real robot", 0.0, 6.0,
                            config.REWARDS["imitation_weight"], 0.5)

rewards = {"left": {"too_close": rw_close, "ideal": rw_ideal, "far": rw_far, "lost": rw_lost},
           "front": {"near": rw_front, "mid": 0.0, "far": config.REWARDS["front"]["far"]},
           "imitation_weight": rw_imit}


@st.cache_resource(show_spinner="Running value iteration…")
def train_cached(variant, gamma, tol, rewards_json, train_frac):
    return pipeline.train(variant, gamma, tol, json.loads(rewards_json), train_frac)


R = train_cached(variant, gamma, tol, json.dumps(rewards, sort_keys=True), train_frac)
mdp, V, policy, Q = R["mdp"], R["V"], R["policy"], R["Q"]
observed = int((mdp.visits > 0).sum())

# ---------------------------------------------------------------- downloads (sidebar)
vdf, pdf = value_df(mdp, V), policy_df(mdp, policy, Q)
with st.sidebar:
    st.markdown("### Outputs")
    st.download_button("Download optimal_value_function.csv", vdf.to_csv(index=False),
                       "optimal_value_function.csv", "text/csv", width="stretch")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("optimal_value_function.csv", vdf.to_csv(index=False))
        z.writestr("optimal_policy.csv", pdf.to_csv(index=False))
        z.writestr("transition_probabilities.csv", mdp.transitions_df().to_csv(index=False))
        z.writestr("rewards.csv", mdp.rewards_df().to_csv(index=False))
        z.writestr("convergence_history.csv", pd.DataFrame(
            {"iteration": range(1, len(R["history"]) + 1), "max_delta": R["history"]}).to_csv(index=False))
        z.writestr("evaluation_report.txt",
                   f"variant={variant} gamma={gamma} tolerance={tol} iterations={len(R['history'])}\n"
                   f"{mdp.summary()}\nAgreement train={R['acc_train']:.4f} test={R['acc_test']:.4f}\n\n"
                   + R["report"].to_string(index=False) + "\n\n" + R["confusion"].to_string())
    st.download_button("Download all results (.zip)", buf.getvalue(), "mdp_results.zip",
                       "application/zip", width="stretch")

# ---------------------------------------------------------------- train the other three algorithms
@st.cache_resource(show_spinner=False)
def train_models(variant, gamma, tol, rewards_json, train_frac):
    r = train_cached(variant, gamma, tol, rewards_json, train_frac)
    bar = st.progress(0.0, text="Preparing the algorithms…")
    models = compare.train_all(r, json.loads(rewards_json), progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    return models


@st.cache_data(show_spinner="Driving every robot through the hall…")
def mission_runs(_models, key, noise, seed):
    return compare.missions(_models, noise=noise, seed=seed)


@st.cache_data(show_spinner="Running 10 noisy missions per algorithm…")
def robustness_table(_models, key):
    return compare.robustness(_models, runs=10)


rj = json.dumps(rewards, sort_keys=True)
MKEY = (variant, gamma, tol, rj, train_frac)
models = train_models(*MKEY)

# ---------------------------------------------------------------- header
chips = "".join(f'<span class="chip"><i style="background:{A.COLORS[k]}"></i>{A.NAMES[k]}</span>'
                for k in ("vi", "mc", "hj", "ga"))
st.markdown(f"""
<div class="hero">
  {robot_svg(88)}
  <div>
    <h1>Wall-following robot navigator</h1>
    <p>5,456 sonar readings from a SCITOS G5 robot, turned into a Markov decision process and solved with
    value iteration, then raced against three other methods through the same hall.</p>
    <div class="chips">{chips}</div>
  </div>
  <div class="stats">
    <div class="stat" style="--c:#2F6FDB"><b>{len(R['history'])}</b><span>value iteration sweeps</span></div>
    <div class="stat" style="--c:#139A6B"><b>{R['acc_test']:.1%}</b><span>match with real robot</span></div>
    <div class="stat" style="--c:#E07A10"><b>{sum(v['metrics']['completed'] for v in mission_runs(models, MKEY, False, 0).values())}/4</b><span>robots reach the exit</span></div>
  </div>
</div>""", unsafe_allow_html=True)

tab_live, tab_alg, tab_vi, tab_pol, tab_eval, tab_data = st.tabs(
    [":material/route: Mission", ":material/model_training: Algorithms", ":material/functions: Value iteration",
     ":material/grid_view: Policy explorer", ":material/fact_check: Evaluation", ":material/database: Dataset"])


def show_iframe(html, height):
    if hasattr(st, "iframe"):                       # Streamlit >= 1.52
        st.iframe(html, height=height)
    else:                                           # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=True)


# ================================================================ MISSION
with tab_live:
    c1, c2, _ = st.columns([1.3, 1.2, 2.2])
    noise = c1.toggle("Sensor noise", value=False, help="Adds ±3 cm sonar noise and small steering and "
                                                        "speed errors, like a real robot.")
    seed = c2.number_input("Noise seed", 0, 999, 0, disabled=not noise,
                           help="Each seed is a different random draw of the noise.")
    runs = mission_runs(models, MKEY, noise, int(seed) if noise else 0)
    show_iframe(compare.view_html(runs), 750)

    st.markdown("#### This run")
    best = min((v["metrics"]["steps"], k) for k, v in runs.items() if v["metrics"]["completed"])[1] \
        if any(v["metrics"]["completed"] for v in runs.values()) else None
    cards = []
    for k, v in runs.items():
        m = v["metrics"]
        badge = '<span class="badge ok">Reached exit</span>' if m["completed"] else '<span class="badge no">Did not finish</span>'
        fastest = ' <span class="badge" style="background:#FFF4D6;color:#8A5A00">Fastest</span>' if k == best else ""
        cards.append(f"""<div class="rc" style="--c:{A.COLORS[k]}">
          <div class="hd">{robot_svg(30, A.COLORS[k])}<b>{A.NAMES[k]}</b></div>
          <div>{badge}{fastest}</div>
          <div class="big" style="margin-top:10px"><div><b>{m['steps']}</b><span>steps ({m['steps'] / 9:.0f} s at 9 Hz)</span></div>
            <div><b>{float(m['distance_m']):.1f} m</b><span>path</span></div></div>
          <div class="row"><span>At ideal wall distance</span><b>{m['ideal_band_pct']:.0f}%</b></div>
          <div class="meter"><i style="width:{m['ideal_band_pct']}%"></i></div>
          <div class="row"><span>Collisions</span><b>{m['collisions']}</b></div>
          <div class="row"><span>Checkpoints</span><b>{m['checkpoints']}/{len(world.CHECKPOINTS)}</b></div>
          <div class="row"><span>Fitness</span><b>{m['fitness']:.0f}</b></div></div>""")
    st.markdown(f'<div class="results">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.markdown(
        "<p class='explain'>Every robot starts in the entry corridor, keeps the wall on its left like the "
        "real SCITOS G5, passes the recessed bay and the wall column, and leaves through the exit. The four "
        "runs are simulated in Python with identical sensors, actions and physics, and the view replays them "
        "with the robots entering one after another. Fitness = 100 per checkpoint + 2 × ideal-distance % "
        "− 10 per collision + 1000 for the exit − 0.5 per step.</p>", unsafe_allow_html=True)

# ================================================================ ALGORITHMS
with tab_alg:
    st.markdown("#### Reliability over 10 noisy missions")
    rob = robustness_table(models, MKEY)
    st.dataframe(rob, hide_index=True, width="stretch")

    agree = compare.dataset_agreement(models, R["test_df"])
    fig = go.Figure(go.Bar(x=[A.NAMES[k].replace(" ", "<br>", 1) for k in agree], y=[v * 100 for v in agree.values()],
                           marker_color=[A.COLORS[k] for k in agree], text=[f"{v:.1%}" for v in agree.values()],
                           textposition="outside"))
    fig.update_yaxes(title="Same action as the real robot (%)", range=[0, 110])
    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(style_fig(fig, 330, title="Agreement with the recorded robot (test data)"),
                        width="stretch")
    with right:
        st.markdown(
            "<p class='explain'>Only value iteration learns from the recording itself, so it copies the real "
            "robot most closely. The other three learn from the simulated hall, where many different action "
            "choices also get the job done. A low agreement score here does not mean a worse robot: all four "
            "reach the exit. It shows how many different policies can solve the same task.</p>",
            unsafe_allow_html=True)

    def algo_header(k, text):
        st.markdown(f"<h4 style='border-left:5px solid {A.COLORS[k]};padding-left:10px;margin-top:18px'>"
                    f"{A.NAMES[k]}</h4><p class='explain'>{text}</p>", unsafe_allow_html=True)

    # --- value iteration
    algo_header("vi", "Model-based. The transition probabilities P(s′|s,a) are counted from the time-ordered "
                      "dataset, the rewards come from the sidebar, and Bellman updates run until no state "
                      f"value changes by more than {tol:.0e}. It never sees the hall: the policy comes "
                      "entirely from the recorded robot.")
    h = R["history"]
    fig = go.Figure(go.Scatter(x=list(range(1, len(h) + 1)), y=h, mode="lines", line=dict(color=A.COLORS["vi"], width=2.5)))
    fig.update_yaxes(type="log", title="max |ΔV|"); fig.update_xaxes(title="Iteration")
    st.plotly_chart(style_fig(fig, 260), width="stretch")

    # --- monte carlo
    mc = models["mc"]["details"]
    algo_header("mc", f"Model-free reinforcement learning. The robot drives {len(mc['history'])} full episodes "
                      "through the hall with an ε-greedy policy (ε falls from 0.4 to 0.02). After each episode, "
                      "first-visit Monte Carlo updates Q(s,a) with the average discounted return (γ = 0.97) that "
                      "followed each state-action pair. Half the episodes use exploring starts, beginning at a "
                      "random point on the route so the late turns, like the one into the exit, get practised. "
                      "Rewards: the same wall-distance rewards as the MDP, +20 per checkpoint, +300 for the exit, "
                      "−25 per collision. Three independent training runs are made and the one that scores best "
                      "on separate validation missions is kept.")
    st.dataframe(pd.DataFrame(mc["all_runs"]), hide_index=True, width="content")
    hm = pd.DataFrame(mc["history"])
    hm["return (20-episode mean)"] = hm["return"].rolling(20, min_periods=1).mean()
    hm["completion rate (20-episode)"] = hm["completed"].astype(float).rolling(20, min_periods=1).mean() * 100
    fig = go.Figure()
    fig.add_scatter(x=hm["episode"], y=hm["return"], mode="markers", name="episode return (kept run)",
                    marker=dict(color=A.COLORS["mc"], size=4, opacity=.35))
    fig.add_scatter(x=hm["episode"], y=hm["return (20-episode mean)"], mode="lines", name="20-episode mean",
                    line=dict(color=A.COLORS["mc"], width=3))
    fig.add_scatter(x=hm["episode"], y=hm["completion rate (20-episode)"], mode="lines", name="reached exit (%)",
                    yaxis="y2", line=dict(color=INK, width=2, dash="dot"))
    fig.update_layout(yaxis=dict(title="Return"), yaxis2=dict(title="Reached exit (%)", overlaying="y",
                                                               side="right", range=[0, 105], showgrid=False,
                                                               tickvals=[0, 25, 50, 75, 100]))
    fig.update_xaxes(title="Episode")
    st.plotly_chart(style_fig(fig, 300, title="Learning curve"), width="stretch")

    # --- hooke-jeeves
    hj = models["hj"]["details"]
    algo_header("hj", "Derivative-free pattern search. A simple wall-following rule has three distance "
                      "thresholds. Hooke-Jeeves probes each threshold up and down (exploratory move), jumps "
                      "further in any direction that improved the score (pattern move), and halves its step "
                      f"size when nothing improves. It used {hj['evaluations']} simulated missions, each scored "
                      "on one clean and two noisy runs.")
    hh = pd.DataFrame(hj["history"])
    left, right = st.columns([1.3, 1])
    with left:
        fig = go.Figure(go.Scatter(x=hh["evaluation"], y=hh["best_fitness"], mode="lines+markers",
                                   line=dict(color=A.COLORS["hj"], width=2.5, shape="hv")))
        fig.update_xaxes(title="Simulated missions"); fig.update_yaxes(title="Best fitness")
        st.plotly_chart(style_fig(fig, 280, title="Search progress"), width="stretch")
    with right:
        th0 = hh["theta"].iloc[0]
        st.dataframe(pd.DataFrame({"threshold": A.HJ_LABELS, "start (m)": th0,
                                   "found (m)": np.round(hj["theta"], 3)}), hide_index=True, width="stretch")
        st.markdown("<p class='explain'>The start values fail the mission. The thresholds it finds sit close "
                    "to the bins derived from the dataset (front 0.9 m, left 0.5 to 0.9 m), found without "
                    "ever seeing the data.</p>", unsafe_allow_html=True)

    # --- genetic algorithm
    ga = models["ga"]["details"]
    algo_header("ga", f"Evolutionary search over whole policies. Each chromosome holds one action per MDP state "
                      f"({len(ga['table'])} genes). A population of 24 evolves for 30 generations with "
                      "tournament selection, uniform crossover, mutation and 2 elites. Each candidate is scored "
                      "on one clean and three noisy missions, which stops evolution from rewarding policies "
                      "that only work in a perfect simulator.")
    hg = pd.DataFrame(ga["history"])
    fig = go.Figure()
    fig.add_scatter(x=hg["generation"], y=hg["best"], name="best", line=dict(color=A.COLORS["ga"], width=3))
    fig.add_scatter(x=hg["generation"], y=hg["mean"], name="population mean", line=dict(color=A.COLORS["ga"], width=2, dash="dash"))
    fig.update_xaxes(title="Generation"); fig.update_yaxes(title="Fitness")
    st.plotly_chart(style_fig(fig, 280, title="Evolution"), width="stretch")

    # --- policies side by side
    st.markdown("#### What each table policy does in the most common states")
    sp = models["vi"]["controller"].space
    top = np.argsort(-mdp.visits)[:12]
    short = {"Move-Forward": "↑ forward", "Slight-Right-Turn": "↗ slight right",
             "Sharp-Right-Turn": "↱ sharp right", "Slight-Left-Turn": "↖ slight left"}
    tbl = {"state": [sp.name(i) for i in top], "times in data": mdp.visits[top]}
    for k in ("vi", "mc", "ga"):
        c = models[k]["controller"]
        tbl[A.NAMES[k]] = [short[config.ACTIONS[c.table[c.resolve[i]]]] for i in top]
    st.dataframe(pd.DataFrame(tbl), hide_index=True, width="stretch")

# ================================================================ VALUE ITERATION
with tab_vi:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iterations", len(R["history"]))
    c2.metric("Final max |ΔV|", f"{R['history'][-1]:.1e}")
    c3.metric("Solve time", f"{R['vi_ms']:.1f} ms")
    c4.metric("Discount γ", f"{gamma:.2f}")

    st.latex(r"V_{k+1}(s) = \max_a \Big[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V_k(s') \Big]"
             r"\qquad \text{stop when } \max_s |V_{k+1}(s) - V_k(s)| < \varepsilon")

    left, right = st.columns([1.1, 1])
    with left:
        h = R["history"]
        fig = go.Figure(go.Scatter(x=list(range(1, len(h) + 1)), y=h, mode="lines",
                                   line=dict(color=A.COLORS["vi"], width=3), name="max |ΔV|",
                                   fill="tozeroy", fillcolor="rgba(47,111,219,0.08)"))
        fig.add_hline(y=tol, line_dash="dash", line_color=ACOL["Sharp-Right-Turn"])
        fig.add_annotation(xref="paper", x=0.03, y=float(np.log10(tol)), yref="y", yanchor="bottom",
                           xanchor="left", showarrow=False, text=f"threshold {tol:.0e}",
                           font=dict(color=ACOL["Sharp-Right-Turn"]))   # log axis: y is given as log10
        fig.update_yaxes(type="log", title="max |ΔV| (log scale)")
        fig.update_xaxes(title="Iteration")
        st.plotly_chart(style_fig(fig, 380, title="Convergence"), width="stretch")
    with right:
        order = np.argsort(-V)
        seen = mdp.visits[order] > 0
        fig = go.Figure()
        for a in mdp.actions:
            m = np.array([mdp.actions[policy[i]] == a for i in order])
            fig.add_bar(x=np.arange(mdp.nS)[m], y=V[order][m], name=ANAME[a], marker_color=ACOL[a],
                        marker_opacity=np.where(seen[m], 1.0, 0.35),
                        customdata=[mdp.space.name(i) for i in order[m]],
                        hovertemplate="%{customdata}<br>V* = %{y:.2f}<extra></extra>")
        fig.update_xaxes(title="States, sorted by value (faded = never seen in data)", showticklabels=False)
        fig.update_yaxes(title="V*(s)")
        st.plotly_chart(style_fig(fig, 380, title="Optimal value per state", legend_cols=2, barmode="overlay"),
                        width="stretch")

    st.markdown("#### optimal_value_function.csv")
    st.dataframe(vdf.assign(visits=mdp.visits, optimal_action=[ANAME[mdp.actions[a]] for a in policy]),
                 width="stretch", height=320, hide_index=True,
                 column_config={"optimal_value": st.column_config.ProgressColumn(
                     "optimal_value", format="%.3f", min_value=float(V.min()), max_value=float(V.max()))})

# ================================================================ POLICY EXPLORER
with tab_pol:
    st.markdown("#### What the robot does in each situation")
    fb, lb = config.BINS["SD_front"][1], config.BINS["SD_left"][1]
    fixed = {}
    others = [s for s in R["sensors"] if s not in ("SD_front", "SD_left")]
    if others:
        cols = st.columns(len(others) + 2)
        labs = pd.DataFrame([mdp.space.labels(i) for i in mdp.state_seq])
        for c, s in zip(cols, others):
            opts = config.BINS[s][1]
            fixed[s] = c.selectbox(f"{s.replace('SD_', '').title()} distance", opts,
                                   index=opts.index(labs[s].mode()[0]))
    z, text, hover = [], [], []
    short = {"Move-Forward": "↑ forward", "Slight-Right-Turn": "↗ slight right",
             "Sharp-Right-Turn": "↱ sharp right", "Slight-Left-Turn": "↖ slight left"}
    for f in fb:
        zr, tr, hr = [], [], []
        for l in lb:
            key = tuple({"SD_front": f, "SD_left": l, **fixed}[s] for s in R["sensors"])
            i = mdp.space.index[key]
            a = mdp.actions[policy[i]]
            zr.append(V[i]); tr.append(f"<b>{short[a]}</b><br>V* {V[i]:.1f}"
                                       + ("" if mdp.visits[i] else "<br><i>unseen</i>"))
            hr.append(f"state {i}: {mdp.space.name(i)}<br>visits {mdp.visits[i]}")
        z.append(zr); text.append(tr); hover.append(hr)
    act_idx = [[mdp.actions.index(next(a for a in mdp.actions if short[a] in t)) for t in row] for row in text]
    cs = []
    for j, a in enumerate(mdp.actions):
        cs += [[j / 4, ACOL[a]], [(j + 1) / 4, ACOL[a]]]
    fig = go.Figure(go.Heatmap(z=act_idx, x=[l.replace("_", " ") for l in lb], y=[f for f in fb], text=text,
                               texttemplate="%{text}", textfont=dict(color="white", size=13),
                               customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                               colorscale=cs, zmin=-0.5, zmax=3.5, showscale=False, xgap=3, ygap=3))
    fig.update_xaxes(title="Left wall distance", side="bottom")
    fig.update_yaxes(title="Front distance", autorange="reversed")
    st.plotly_chart(style_fig(fig, 360), width="stretch")

    st.markdown("#### Transition explorer")
    st.markdown("<p class='explain'>Pick a state and an action to see where the robot ended up next in the "
                "recorded data. These counts are the transition probabilities P(s′ | s, a).</p>",
                unsafe_allow_html=True)
    seen_states = [i for i in np.argsort(-mdp.visits) if mdp.visits[i] > 0]
    c1, c2 = st.columns([2, 1])
    s_sel = c1.selectbox("State", seen_states,
                         format_func=lambda i: f"{i}: {mdp.space.name(i)}  ({mdp.visits[i]} visits)")
    a_sel = c2.selectbox("Action", mdp.actions, format_func=ANAME.get,
                         index=int(policy[s_sel]))
    j = mdp.actions.index(a_sel)
    probs = mdp.P[s_sel, j]
    nz = np.argsort(-probs)[: min(10, int((probs > 0).sum()))]
    n_obs = int(mdp.transition_counts[s_sel, j].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Times this action was taken here", n_obs)
    c2.metric("R(s, a)", f"{mdp.R[s_sel, j]:.2f}")
    c3.metric("Q(s, a)", f"{Q[s_sel, j]:.2f}")
    if n_obs == 0:
        st.info("The real robot never took this action in this state, so the state's overall "
                "next-state distribution is used instead.")
    fig = go.Figure(go.Bar(y=[mdp.space.name(k) for k in nz][::-1], x=probs[nz][::-1], orientation="h",
                           marker_color=ACOL[a_sel], text=[f"{p:.1%}" for p in probs[nz][::-1]],
                           textposition="outside"))
    fig.update_xaxes(title="P(s′ | s, a)", range=[0, 1.12], tickformat=".0%")
    st.plotly_chart(style_fig(fig, 60 + 34 * len(nz)), width="stretch")

# ================================================================ EVALUATION
with tab_eval:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agreement, training data", f"{R['acc_train']:.1%}")
    c2.metric("Agreement, unseen test data", f"{R['acc_test']:.1%}")
    c3.metric("Test rows", f"{len(R['test_df']):,}")
    c4.metric("Macro F1 (test)", f"{R['report']['f1'].mean():.3f}")

    left, right = st.columns([1, 1])
    with left:
        cm = R["confusion"]
        norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        fig = go.Figure(go.Heatmap(
            z=norm.values, x=[ANAME[a].replace(" ", "<br>", 1) for a in cm.columns], y=[ANAME[a] for a in cm.index],
            text=[[f"{v}<br>{p:.0%}" for v, p in zip(r1, r2)] for r1, r2 in zip(cm.values, norm.values)],
            texttemplate="%{text}", colorscale=[[0, "#F4F8FD"], [0.5, "#7FA8EC"], [1, "#1E4FA8"]],
            showscale=False, xgap=2, ygap=2))
        fig.update_xaxes(title="Policy chose"); fig.update_yaxes(title="Real robot did", autorange="reversed")
        st.plotly_chart(style_fig(fig, 400, title="Confusion matrix (test data)"), width="stretch")
    with right:
        st.markdown("##### Per-action results (test data)")
        rep = R["report"].copy(); rep["class"] = rep["class"].map(ANAME)
        st.dataframe(rep, hide_index=True, width="stretch")
        st.markdown(
            "<p class='explain'>Agreement measures how often the value-iteration policy chooses the same "
            "action the real robot chose for the same readings. Mistakes concentrate on slight turns, "
            "where the discretised state cannot always tell a gentle correction from going straight.</p>",
            unsafe_allow_html=True)

    st.markdown("#### How settings change the result")

    @st.cache_data(show_spinner="Sweeping γ…")
    def gamma_sweep(variant, tol, rewards_json, train_frac):
        rows = []
        for g in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            r = pipeline.train(variant, g, tol, json.loads(rewards_json), train_frac)
            rows.append({"gamma": g, "iterations": len(r["history"]), "test agreement": r["acc_test"]})
        return pd.DataFrame(rows)

    @st.cache_data(show_spinner="Sweeping threshold…")
    def tol_sweep(variant, gamma, rewards_json, train_frac):
        ref = pipeline.train(variant, gamma, 1e-10, json.loads(rewards_json), train_frac)["V"]
        rows = []
        for t in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            r = pipeline.train(variant, gamma, t, json.loads(rewards_json), train_frac)
            rows.append({"tolerance": t, "iterations": len(r["history"]),
                         "max error vs exact V*": float(np.max(np.abs(r["V"] - ref)))})
        return pd.DataFrame(rows)

    gs, ts = gamma_sweep(variant, tol, rj, train_frac), tol_sweep(variant, gamma, rj, train_frac)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=gs["gamma"], y=gs["iterations"], mode="lines+markers", name="iterations",
                        line=dict(color=INK, width=2.5))
        fig.add_scatter(x=gs["gamma"], y=gs["test agreement"] * 100, mode="lines+markers",
                        name="test agreement %", yaxis="y2", line=dict(color=ACOL["Move-Forward"], width=2.5))
        fig.update_layout(yaxis=dict(title="Iterations"),
                          yaxis2=dict(title="Agreement %", overlaying="y", side="right", showgrid=False,
                                      tickformat=".1f"))
        fig.update_xaxes(title="Discount factor γ")
        st.plotly_chart(style_fig(fig, 340, title="Discount factor"), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=ts["tolerance"], y=ts["max error vs exact V*"], mode="lines+markers",
                        name="max error of V", line=dict(color=ACOL["Sharp-Right-Turn"], width=2.5))
        fig.add_scatter(x=ts["tolerance"], y=ts["iterations"], mode="lines+markers", name="iterations",
                        yaxis="y2", line=dict(color=INK, width=2.5))
        fig.update_layout(yaxis=dict(title="Max error vs exact V*", type="log"),
                          yaxis2=dict(title="Iterations", overlaying="y", side="right", showgrid=False,
                                      tickformat=".0f"))
        fig.update_xaxes(title="Convergence threshold", type="log", autorange="reversed")
        st.plotly_chart(style_fig(fig, 340, title="Accuracy depends on the threshold"),
                        width="stretch")


# ================================================================ DATASET
with tab_data:
    df = load_dataset("4")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded steps", f"{len(df):,}")
    c2.metric("Sampling rate", "9 Hz")
    c3.metric("Laps of the room", "4")
    c4.metric("Sonar sensors", "24")

    left, right = st.columns([1, 1.4])
    with left:
        vc = df["Class"].value_counts().reindex(config.ACTIONS)
        fig = go.Figure(go.Bar(x=[ANAME[a] for a in vc.index], y=vc.values,
                               marker_color=[ACOL[a] for a in vc.index], text=vc.values, textposition="outside"))
        fig.update_yaxes(title="Rows", range=[0, vc.max() * 1.15])
        st.plotly_chart(style_fig(fig, 360, title="Actions taken by the real robot"), width="stretch")
    with right:
        fig = go.Figure()
        for s in config.ALL_SENSORS:
            for a in config.ACTIONS:
                fig.add_box(y=df.loc[df["Class"] == a, s], x=[s.replace("SD_", "")] * int((df["Class"] == a).sum()),
                            name=ANAME[a], marker_color=ACOL[a], legendgroup=a,
                            showlegend=(s == "SD_front"), boxpoints=False)
        fig.update_yaxes(title="Distance (m)", range=[0, 3.5])
        st.plotly_chart(style_fig(fig, 360, title="Sensor readings by action", legend_cols=2, boxmode="group"),
                        width="stretch")

    st.markdown("#### Recording timeline")
    start = st.slider("Start step", 0, len(df) - 300, 0, 50)
    w = df.iloc[start:start + 300]
    fig = go.Figure()
    for s in config.ALL_SENSORS:
        fig.add_scatter(x=w.index, y=w[s], mode="lines", name=s, line=dict(color=SCOL[s], width=2))
    for a in config.ACTIONS:
        m = w["Class"] == a
        fig.add_scatter(x=w.index[m], y=[-0.15] * int(m.sum()), mode="markers", name=ANAME[a],
                        marker=dict(color=ACOL[a], symbol="square", size=7))
    fig.update_yaxes(title="Distance (m)", range=[-0.35, 3.2]); fig.update_xaxes(title="Step")
    st.plotly_chart(style_fig(fig, 360, legend_cols=4), width="stretch")

    st.markdown("#### The 24 raw sonars")
    raw = load_raw24()
    match = verify_24_to_4()
    left, right = st.columns([1, 1.2])
    with left:
        k = st.slider("Step to inspect", 0, len(raw) - 1, 120)
        vals = raw.iloc[k, :24].to_numpy(float)
        angles = np.arange(24) * 15.0   # sensor 1 at front, numbered clockwise
        fig = go.Figure()
        colors = ["#B7C2CD"] * 24
        for s, cols in config.ARC_COLUMNS_24.items():
            for c in cols:
                colors[c] = SCOL[s]
        fig.add_barpolar(r=vals, theta=angles, width=[13] * 24, marker_color=colors,
                         customdata=[f"US{i + 1}" for i in range(24)],
                         hovertemplate="%{customdata}: %{r:.2f} m<extra></extra>")
        fig.update_layout(polar=dict(angularaxis=dict(rotation=90, direction="clockwise", tickvals=[0, 90, 180, 270],
                                                      ticktext=["front", "right", "back", "left"]),
                                     radialaxis=dict(range=[0, 5], ticksuffix=" m", angle=45, tickangle=45,
                                                     tickfont=dict(size=10, color=MUTED))),
                          showlegend=False)
        st.plotly_chart(style_fig(fig, 380, title=f"Step {k}: {ANAME[raw['Class'].iloc[k]]}"),
                        width="stretch")
    with right:
        red = {s: raw.iloc[k, cols].astype(float).min() for s, cols in config.ARC_COLUMNS_24.items()}
        st.dataframe(pd.DataFrame({
            "simplified distance": list(red),
            "raw sonar columns": [", ".join(f"US{c + 1}" for c in config.ARC_COLUMNS_24[s]) for s in red],
            "min of raw sonars (m)": [round(v, 3) for v in red.values()],
            "value in 4-sensor file (m)": [round(df.loc[k, s], 3) for s in red],
            "rows matching (all data)": [f"{match[s]:.0%}" for s in red]}), hide_index=True,
            width="stretch")
        st.markdown(
            "<p class='explain'>The four simplified distances used by the MDP are the minimum of the raw "
            "sonars inside each 60° arc. Recomputing them from the 24-sensor file reproduces the 4-sensor "
            "file exactly for every row, which confirms the three files describe the same recording.</p>",
            unsafe_allow_html=True)
