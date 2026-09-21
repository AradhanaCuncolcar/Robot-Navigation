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

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Semi+Condensed:wght@500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea, label {{
  font-family: "Barlow", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important; }}
.block-container {{ padding-top: 1.6rem; max-width: 1480px; }}
h1, h2, h3, h4 {{ font-family: "Barlow Semi Condensed", "Barlow", sans-serif !important;
  color: {INK}; letter-spacing: 0; }}
[data-testid="stMetricValue"] {{ font-family: "Barlow Semi Condensed", sans-serif; font-weight: 700;
  font-variant-numeric: tabular-nums; color: {INK}; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.9rem; }}
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px;
  padding: 12px 16px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 1rem; padding: 8px 16px; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
.hero {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 24px;
  flex-wrap: wrap; margin-bottom: 6px; }}
.hero h1 {{ font-size: 2.35rem; line-height: 1.05; margin: 0; padding: 0; font-weight: 700; }}
.hero p {{ color: {MUTED}; margin: 6px 0 0; max-width: 70ch; font-size: 1.02rem; }}
.status {{ display: flex; gap: 22px; flex-wrap: wrap; }}
.status div {{ border-left: 3px solid {INK}; padding-left: 10px; }}
.status b {{ display: block; font-family: "Barlow Semi Condensed", sans-serif; font-size: 1.5rem;
  font-variant-numeric: tabular-nums; color: {INK}; line-height: 1.1; }}
.status span {{ color: {MUTED}; font-size: .85rem; }}
.explain {{ color: #44525F; max-width: 78ch; }}
</style>""", unsafe_allow_html=True)


def style_fig(fig, height=360, **kw):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Barlow, sans-serif", size=13, color=INK),
        title_font=dict(family="Barlow Semi Condensed, sans-serif", size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(gridcolor="#EDF0F3", zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor="#EDF0F3", zerolinecolor=LINE, linecolor=LINE)
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
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
st.markdown(f"""
<div class="hero">
  <div>
    <h1>Wall-following robot navigator</h1>
    <p>A SCITOS G5 robot's 5,456 recorded sonar readings, turned into a Markov decision process and
    solved with value iteration, then raced against Monte Carlo control, Hooke-Jeeves pattern search
    and a genetic algorithm in the same hall.</p>
  </div>
  <div class="status">
    <div><b>{len(R['history'])}</b><span>value iteration sweeps</span></div>
    <div><b>{R['acc_test']:.1%}</b><span>match with real robot (test)</span></div>
    <div><b>4</b><span>algorithms, one room</span></div>
  </div>
</div>""", unsafe_allow_html=True)

tab_live, tab_alg, tab_vi, tab_pol, tab_eval, tab_data = st.tabs(
    ["Mission", "Algorithms", "Value iteration", "Policy explorer", "Evaluation", "Dataset"])


def show_iframe(html, height):
    if hasattr(st, "iframe"):                       # Streamlit >= 1.52
        st.iframe(html, height=height)
    else:                                           # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=True)


# ================================================================ MISSION
with tab_live:
    c1, c2, c3, _ = st.columns([1.2, 1, 1.2, 2.5])
    start_view = c1.radio("Open in", ["Floor plan", "3D view"], horizontal=True)
    noise = c2.toggle("Sensor noise", value=False, help="Adds ±3 cm sonar noise and small steering and "
                                                        "speed errors, like a real robot.")
    seed = c3.number_input("Noise seed", 0, 999, 0, disabled=not noise,
                           help="Each seed is a different random draw of the noise.")
    runs = mission_runs(models, MKEY, noise, int(seed) if noise else 0)
    show_iframe(compare.view_html(runs, "3d" if start_view == "3D view" else "2d"), 790)

    rows = []
    for k, v in runs.items():
        m = v["metrics"]
        rows.append({"algorithm": A.NAMES[k], "reached exit": "Yes" if m["completed"] else "No",
                     "steps": m["steps"], "time at 9 Hz (s)": round(m["steps"] / 9, 1),
                     "path length (m)": float(m["distance_m"]), "collisions": m["collisions"],
                     "checkpoints": f"{m['checkpoints']}/{len(world.CHECKPOINTS)}",
                     "at ideal wall distance (%)": m["ideal_band_pct"], "fitness": m["fitness"]})
    st.markdown("#### This run")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"fitness": st.column_config.NumberColumn(
                     help="100 per checkpoint + 2 × ideal-distance % − 10 per collision + "
                          "1000 for reaching the exit − 0.5 per step")})
    st.markdown(
        "<p class='explain'>Every robot starts in the entry corridor, keeps the wall on its left like the "
        "real SCITOS G5, passes the recessed bay and the wall column, and leaves through the exit. The "
        "pillars stay in the middle of the hall, out of the robot's way. The four runs are simulated in "
        "Python with identical sensors, actions and physics; the view replays them step by step, with the "
        "robots entering one after another.</p>", unsafe_allow_html=True)

# ================================================================ ALGORITHMS
with tab_alg:
    st.markdown("#### Reliability over 10 noisy missions")
    rob = robustness_table(models, MKEY)
    st.dataframe(rob, hide_index=True, width="stretch")

    agree = compare.dataset_agreement(models, R["test_df"])
    fig = go.Figure(go.Bar(x=[A.NAMES[k] for k in agree], y=[v * 100 for v in agree.values()],
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
                                                               side="right", range=[0, 105], showgrid=False))
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
                                   line=dict(color=INK, width=2.5), name="max |ΔV|"))
        fig.add_hline(y=tol, line_dash="dash", line_color=ACOL["Sharp-Right-Turn"],
                      annotation_text=f"threshold {tol:.0e}", annotation_position="top right")
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
        st.plotly_chart(style_fig(fig, 380, title="Optimal value per state", barmode="overlay"),
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
    fig = go.Figure(go.Heatmap(z=z, x=[l.replace("_", " ") for l in lb], y=[f for f in fb], text=text,
                               texttemplate="%{text}", customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                               colorscale=[[0, "#E7ECF1"], [1, "#6F8FAF"]], colorbar=dict(title="V*")))
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
            z=norm.values, x=[ANAME[a] for a in cm.columns], y=[ANAME[a] for a in cm.index],
            text=[[f"{v}<br>{p:.0%}" for v, p in zip(r1, r2)] for r1, r2 in zip(cm.values, norm.values)],
            texttemplate="%{text}", colorscale=[[0, "#FFFFFF"], [1, INK]], showscale=False))
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
                          yaxis2=dict(title="Agreement %", overlaying="y", side="right", showgrid=False))
        fig.update_xaxes(title="Discount factor γ")
        st.plotly_chart(style_fig(fig, 340, title="Discount factor"), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=ts["tolerance"], y=ts["max error vs exact V*"], mode="lines+markers",
                        name="max error of V", line=dict(color=ACOL["Sharp-Right-Turn"], width=2.5))
        fig.add_scatter(x=ts["tolerance"], y=ts["iterations"], mode="lines+markers", name="iterations",
                        yaxis="y2", line=dict(color=INK, width=2.5))
        fig.update_layout(yaxis=dict(title="Max error vs exact V*", type="log"),
                          yaxis2=dict(title="Iterations", overlaying="y", side="right", showgrid=False))
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
        st.plotly_chart(style_fig(fig, 360, title="Sensor readings by action", boxmode="group"),
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
    st.plotly_chart(style_fig(fig, 360), width="stretch")

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
                                     radialaxis=dict(range=[0, 5], ticksuffix=" m")),
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
