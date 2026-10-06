"""
Eco-Friendly Daily Commute Helper -- Simulated Demonstration Engine.

Generates a synthetic respondent-level dataset (N = 120) with a fixed random
seed and computes every statistic reported in the revised paper.  The output is
NOT real data: no human participant provided any value.  The script exists so
that the analytical procedure is fully reproducible.

Outputs
-------
simulated_dataset.csv     respondent-level simulated dataset
analysis_output.json      every statistic used in the paper
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

SEED = 20251006
N = 120
RNG = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# 1. Emission factors (illustrative model inputs, g CO2 per passenger-km)
# --------------------------------------------------------------------------
EF = {
    "C": 170.0,   # private car (solo)
    "B": 70.0,    # city bus (allocated per passenger)
    "M": 30.0,    # metro / rail
    "Y": 0.0,     # bicycle
    "E": 20.0,    # e-bike / e-scooter
    "P": 57.0,    # carpool (car factor / assumed occupancy 3)
    "S": 60.0,    # shared ride (car factor / assumed occupancy ~2.8)
    "X": 42.0,    # metro + e-bike multimodal leg (weighted)
}

MODE_LABEL = {
    "C": "Private car", "B": "City bus", "M": "Metro/rail", "Y": "Bicycle",
    "E": "E-bike", "P": "Carpool", "S": "Shared ride", "X": "Metro + e-bike",
}

# --------------------------------------------------------------------------
# 2. Scenario definitions (design values; time min, cost INR, comfort, conv.)
# --------------------------------------------------------------------------
SCENARIOS = {
    "S1": {"distance": 3,  "context": "Short campus-area trip, fair weather, off-peak",
           "options": {"C": (12, 40, 4, 5), "B": (20, 10, 2, 3), "Y": (15, 0, 2, 3),
                       "E": (11, 25, 3, 4), "S": (13, 30, 3, 4)}},
    "S2": {"distance": 6,  "context": "Peak-hour trip, congested corridor",
           "options": {"C": (25, 85, 4, 5), "B": (35, 15, 2, 3), "Y": (28, 0, 2, 3),
                       "P": (27, 30, 3, 3), "E": (19, 45, 3, 4)}},
    "S3": {"distance": 8,  "context": "Hot, humid day, medium distance",
           "options": {"C": (22, 110, 4, 5), "B": (38, 12, 2, 3), "M": (28, 25, 4, 4),
                       "P": (25, 40, 3, 3), "S": (30, 70, 3, 4)}},
    "S4": {"distance": 12, "context": "Rainy day, longer distance",
           "options": {"C": (35, 170, 4, 5), "B": (50, 15, 2, 3), "M": (38, 40, 4, 4),
                       "P": (38, 60, 3, 3), "S": (40, 110, 3, 4)}},
    "S5": {"distance": 15, "context": "Long trip with first/last-mile leg (multimodal)",
           "options": {"C": (40, 210, 4, 5), "X": (45, 45, 4, 3), "B": (60, 30, 2, 3),
                       "P": (44, 75, 3, 3), "S": (48, 140, 3, 4)}},
    "S6": {"distance": 5,  "context": "Evening return trip, personal-safety concern",
           "options": {"C": (18, 70, 4, 5), "B": (30, 15, 2, 3), "P": (20, 30, 3, 3),
                       "S": (22, 60, 3, 4), "E": (15, 35, 3, 4)}},
}
SCEN_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6"]


def emission(mode: str, distance: float) -> float:
    return distance * EF[mode]


# --------------------------------------------------------------------------
# 3. Demographics and commute profile
# --------------------------------------------------------------------------
def make_demographics(n: int) -> pd.DataFrame:
    gender = RNG.choice(["Male", "Female", "Other/Prefer not to say"],
                        size=n, p=[0.567, 0.408, 0.025])
    year = RNG.choice([1, 2, 3, 4], size=n, p=[0.275, 0.292, 0.308, 0.125])
    programme = RNG.choice(["CSE", "Other engineering", "Management/Commerce", "Sciences/Other"],
                           size=n, p=[0.425, 0.233, 0.217, 0.125])
    age = np.clip(np.round(RNG.normal(19.79, 1.23, n)), 18, 23).astype(int)
    dist_cat = RNG.choice(["<2 km", "2-5 km", "5-10 km", ">10 km"],
                          size=n, p=[0.242, 0.308, 0.250, 0.200])
    days = RNG.choice([3, 4, 5, 6], size=n, p=[0.059, 0.233, 0.408, 0.300])

    # current mode drawn conditional on distance category
    mode_probs = {
        "<2 km":     {"Private car": .06, "Two-wheeler": .12, "Bus": .20, "Walking/Cycling": .48, "Carpool/Shared": .14},
        "2-5 km":    {"Private car": .12, "Two-wheeler": .22, "Bus": .26, "Walking/Cycling": .24, "Carpool/Shared": .16},
        "5-10 km":   {"Private car": .22, "Two-wheeler": .27, "Bus": .28, "Walking/Cycling": .08, "Carpool/Shared": .15},
        ">10 km":    {"Private car": .27, "Two-wheeler": .25, "Bus": .25, "Walking/Cycling": .06, "Carpool/Shared": .17},
    }
    current_mode = [RNG.choice(list(mode_probs[d].keys()), p=list(mode_probs[d].values()))
                    for d in dist_cat]

    return pd.DataFrame({
        "respondent_id": [f"R{ i+1:03d}" for i in range(n)],
        "age": age, "gender": gender, "year_of_study": year, "programme": programme,
        "commute_distance_cat": dist_cat, "current_mode": current_mode,
        "commute_days": days,
    })


# --------------------------------------------------------------------------
# 4. Latent traits -> Likert items
# --------------------------------------------------------------------------
def make_traits(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    pv = df["current_mode"].isin(["Private car", "Two-wheeler"]).astype(float).values
    # correlated latent traits (standardised); modest offsets for private-vehicle users
    mean = np.array([0.10 - 0.05 * pv, 0.16 - 0.02 * pv, 0.18 - 0.06 * pv,
                     -0.15 + 0.18 * pv, -0.05 + 0.10 * pv, 0.02 - 0.05 * pv,
                     0.05 - 0.06 * pv, 0.06 - 0.02 * pv]).T
    corr = np.array([
        # aw    ec    conv  cost  time  inten accept engage
        [1.00, 0.30, -0.05, -0.15, -0.05, 0.45, 0.30, 0.25],
        [0.30, 1.00, -0.05, -0.05, -0.10, 0.32, 0.25, 0.22],
        [-0.05, -0.05, 1.00, 0.05, 0.05, 0.02, 0.50, 0.30],
        [-0.15, -0.05, 0.05, 1.00, -0.05, -0.05, -0.05, -0.05],
        [-0.05, -0.10, 0.05, -0.05, 1.00, -0.12, -0.08, -0.05],
        [0.45, 0.32, 0.02, -0.05, -0.12, 1.00, 0.20, 0.20],
        [0.30, 0.25, 0.50, -0.05, -0.08, 0.20, 1.00, 0.55],
        [0.25, 0.22, 0.30, -0.05, -0.05, 0.20, 0.55, 1.00],
    ])
    L = np.linalg.cholesky(corr)
    z = mean + RNG.standard_normal((n, 8)) @ L.T
    cols = ["t_aw", "t_ec", "t_conv", "t_cost", "t_time", "t_inten", "t_accept", "t_engage"]
    return pd.DataFrame(z, columns=cols)


def likert(latent: np.ndarray, offset: float = 3.5, slope: float = 0.75) -> np.ndarray:
    raw = offset + slope * latent + RNG.normal(0, 0.90, len(latent))
    return np.clip(np.round(raw), 1, 5).astype(int)


def make_items(df: pd.DataFrame) -> pd.DataFrame:
    tr = make_traits(df)
    out = {}
    for i in range(1, 10):
        out[f"aw{i}"] = likert(tr["t_aw"] + RNG.normal(0, 0.25, len(df)))
    for i in range(1, 8):
        out[f"ec{i}"] = likert(tr["t_ec"] + RNG.normal(0, 0.25, len(df)))
    for i in range(1, 4):
        out[f"cost{i}"] = likert(tr["t_cost"] + RNG.normal(0, 0.15, len(df)))
        out[f"time{i}"] = likert(tr["t_time"] + RNG.normal(0, 0.15, len(df)))
        out[f"conv{i}"] = likert(tr["t_conv"] + RNG.normal(0, 0.15, len(df)))
    for name, key in [("comfort", "t_conv"), ("safety", "t_ec"), ("access", "t_conv")]:
        out[name] = likert(tr[key] + RNG.normal(0, 0.6, len(df)))
    for i in range(1, 8):
        out[f"int{ i}"] = likert(tr["t_inten"] + RNG.normal(0, 0.35, len(df)))
    for i in range(1, 8):
        out[f"acc{ i}"] = likert(tr["t_accept"] + RNG.normal(0, 0.30, len(df)))
    # attention check: correct for ~97%
    out["attention_check"] = np.where(RNG.random(len(df)) < 0.97, 4, RNG.integers(1, 6, len(df)))
    out["t_inten"] = tr["t_inten"].values
    out["t_accept"] = tr["t_accept"].values
    out["t_ec"] = tr["t_ec"].values
    out["t_aw"] = tr["t_aw"].values
    out["t_time"] = tr["t_time"].values
    out["t_cost"] = tr["t_cost"].values
    out["t_conv"] = tr["t_conv"].values
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 5. Stage A choices via random-utility (multinomial-logit-type) model
# --------------------------------------------------------------------------
def stage_a_choice(scen: str, df: pd.DataFrame, items: pd.DataFrame) -> str:
    d = SCENARIOS[scen]["distance"]
    opts = SCENARIOS[scen]["options"]
    times = np.array([opts[m][0] for m in opts])
    costs = np.array([opts[m][1] for m in opts])
    comfort = np.array([opts[m][2] for m in opts])
    conv = np.array([opts[m][3] for m in opts])
    emis = np.array([emission(m, d) for m in opts])
    modes = list(opts.keys())

    t_time = items["t_time"].values
    t_cost = items["t_cost"].values
    t_conv = items["t_conv"].values
    t_aw = items["t_aw"].values
    pv = df["current_mode"].isin(["Private car", "Two-wheeler"]).values

    time_z = (times - times.mean()) / times.std()
    cost_z = (costs - costs.mean()) / costs.std()
    com_z = (comfort - comfort.mean()) / comfort.std()
    conv_z = (conv - conv.mean()) / conv.std()
    emi_z = (emis - emis.mean()) / (emis.std() + 1e-9)

    # individual-specific part
    b_time = -0.55 - 0.30 * t_time
    b_cost = -0.45 - 0.35 * t_cost
    b_com = 0.35 + 0.10 * t_conv
    b_conv = 0.40 + 0.20 * t_conv
    b_emi = -0.15 - 0.22 * t_aw
    util = (b_time[:, None] * time_z + b_cost[:, None] * cost_z
            + b_com[:, None] * com_z + b_conv[:, None] * conv_z
            + b_emi[:, None] * emi_z)
    for j, m in enumerate(modes):
        if m == "C":
            util[:, j] += 0.70 * pv.astype(float)
        elif m in ("Y", "E"):
            util[:, j] += 0.25 * (~pv).astype(float)
    util += RNG.gumbel(0, 1, util.shape)
    idx = np.argmax(util, axis=1)
    return [modes[i] for i in idx]


# --------------------------------------------------------------------------
# 6. Personalised recommendation (rank by respondent-specific weights)
# --------------------------------------------------------------------------
def recommend(scen: str, df: pd.DataFrame, items: pd.DataFrame, rows) -> list:
    d = SCENARIOS[scen]["distance"]
    opts = SCENARIOS[scen]["options"]
    modes = list(opts.keys())
    t_time = items["t_time"].values
    t_cost = items["t_cost"].values
    t_aw = items["t_aw"].values

    times = np.array([opts[m][0] for m in modes])
    costs = np.array([opts[m][1] for m in modes])
    emis = np.array([emission(m, d) for m in modes])
    tz = (times - times.mean()) / times.std()
    cz = (costs - costs.mean()) / costs.std()
    ez = (emis - emis.mean()) / (emis.std() + 1e-9)
    wt = 0.9 + 0.4 * t_time
    wc = 0.8 + 0.4 * t_cost
    we = 0.6 + 0.5 * t_aw
    score = -(wt[:, None] * tz + wc[:, None] * cz + we[:, None] * ez)
    idx = np.argmax(score, axis=1)
    return [modes[i] for i in idx]


# --------------------------------------------------------------------------
# 7. Assemble dataset
# --------------------------------------------------------------------------
def build() -> pd.DataFrame:
    df = make_demographics(N)
    items = make_items(df)
    data = pd.concat([df, items], axis=1)

    # scale scores
    data["awareness_score"] = data[[f"aw{i}" for i in range(1, 10)]].mean(axis=1)
    data["concern_score"] = data[[f"ec{i}" for i in range(1, 8)]].mean(axis=1)
    data["cost_sensitivity"] = data[[f"cost{i}" for i in range(1, 4)]].mean(axis=1)
    data["time_sensitivity"] = data[[f"time{i}" for i in range(1, 4)]].mean(axis=1)
    data["convenience_score"] = data[[f"conv{i}" for i in range(1, 4)]].mean(axis=1)
    data["intention_score"] = data[[f"int{i}" for i in range(1, 8)]].mean(axis=1)
    data["acceptance_score"] = data[[f"acc{i}" for i in range(1, 7)]].mean(axis=1)
    data["engagement"] = data[["acc6", "acc7"]].mean(axis=1)
    data["private_vehicle_user"] = data["current_mode"].isin(["Private car", "Two-wheeler"]).astype(int)

    # Stage A
    for s in SCEN_ORDER:
        data[f"stageA_{s}"] = stage_a_choice(s, data, items)
    # recommendation
    for s in SCEN_ORDER:
        data[f"rec_{s}"] = recommend(s, data, items, data)

    # Stage C: follow recommendation with prob depending on acceptance & concern
    acc_z = (data["acceptance_score"] - data["acceptance_score"].mean()) / data["acceptance_score"].std()
    ec_z = (data["t_ec"] - data["t_ec"].mean()) / data["t_ec"].std()
    p_follow = 1 / (1 + np.exp(-(-1.25 + 1.05 * acc_z + 0.15 * ec_z)))
    for s in SCEN_ORDER:
        follow = RNG.random(N) < p_follow.values
        rec = data[f"rec_{s}"].values
        a = data[f"stageA_{s}"].values
        differs = rec != a
        data[f"stageC_{s}"] = np.where(follow & differs, rec, a)

    # emissions
    for s in SCEN_ORDER:
        d = SCENARIOS[s]["distance"]
        data[f"emission_A_{s}"] = [emission(m, d) for m in data[f"stageA_{s}"]]
        data[f"emission_C_{s}"] = [emission(m, d) for m in data[f"stageC_{s}"]]
        data[f"emission_car_{s}"] = emission("C", d)

    data["baseline_emission"] = data[[f"emission_A_{s}" for s in SCEN_ORDER]].mean(axis=1)
    data["recommended_emission"] = data[[f"emission_C_{s}" for s in SCEN_ORDER]].mean(axis=1)
    data["estimated_co2_saved"] = data["baseline_emission"] - data["recommended_emission"]
    data["car_share_A"] = (data[[f"stageA_{s}" for s in SCEN_ORDER]] == "C").mean(axis=1)
    data["car_share_C"] = (data[[f"stageC_{s}" for s in SCEN_ORDER]] == "C").mean(axis=1)
    data["sust_share_A"] = 1 - data["car_share_A"]
    data["sust_share_C"] = 1 - data["car_share_C"]
    data["low_emission_majority"] = (data[[f"stageA_{s}" for s in SCEN_ORDER]] != "C").sum(axis=1).ge(4).astype(int)

    # shift counts
    shift = np.zeros(N, dtype=int)
    for s in SCEN_ORDER:
        shift += ((data[f"stageA_{s}"] == "C") & (data[f"stageC_{s}"] != "C")).astype(int).values
    data["shift_count"] = shift
    data["mode_shift_indicator"] = (shift >= 1).astype(int)
    data["selected_mode"] = data["stageC_S1"]  # representative selected mode
    return data


# --------------------------------------------------------------------------
# 8. Statistics
# --------------------------------------------------------------------------
def cronbach(x: pd.DataFrame) -> float:
    x = x.astype(float)
    k = x.shape[1]
    var_items = x.var(axis=0, ddof=1).sum()
    var_total = x.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - var_items / var_total)


def main():
    data = build()
    res = {}

    # ---- descriptives ----
    res["n"] = int(len(data))
    scales = ["awareness_score", "concern_score", "cost_sensitivity", "time_sensitivity",
              "convenience_score", "intention_score", "acceptance_score", "engagement"]
    res["descriptives"] = {}
    for c in scales:
        x = data[c]
        w = stats.wilcoxon(x - 3) if len(np.unique(x)) > 1 else None
        res["descriptives"][c] = {
            "mean": round(float(x.mean()), 3), "sd": round(float(x.std(ddof=1)), 3),
            "median": round(float(x.median()), 3), "min": round(float(x.min()), 3),
            "max": round(float(x.max()), 3), "skew": round(float(x.skew()), 3),
            "kurt": round(float(x.kurt()), 3),
            "sw_W": round(float(stats.shapiro(x).statistic), 3),
            "sw_p": round(float(stats.shapiro(x).pvalue), 3),
            "wilcoxon_p": round(float(w.pvalue), 4) if w else None,
        }

    # ---- reliability ----
    res["reliability"] = {
        "awareness": round(cronbach(data[[f"aw{i}" for i in range(1, 10)]]), 3),
        "concern": round(cronbach(data[[f"ec{i}" for i in range(1, 8)]]), 3),
        "cost": round(cronbach(data[["cost1", "cost2", "cost3"]]), 3),
        "time": round(cronbach(data[["time1", "time2", "time3"]]), 3),
        "convenience": round(cronbach(data[["conv1", "conv2", "conv3"]]), 3),
        "intention": round(cronbach(data[[f"int{i}" for i in range(1, 8)]]), 3),
        "acceptance": round(cronbach(data[[f"acc{i}" for i in range(1, 7)]]), 3),
    }

    # ---- correlations (Spearman) ----
    corr_cols = ["awareness_score", "concern_score", "intention_score", "convenience_score",
                 "cost_sensitivity", "time_sensitivity", "acceptance_score"]
    res["correlations"] = {}
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            r, p = stats.spearmanr(data[a], data[b])
            res["correlations"][f"{a}|{b}"] = {"rho": round(float(r), 3), "p": round(float(p), 4)}
    r, p = stats.spearmanr(data["concern_score"], data["shift_count"])
    res["correlations"]["concern_score|shift_count"] = {"rho": round(float(r), 3), "p": round(float(p), 4)}

    # ---- group comparisons ----
    def welch(a, b):
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        df = (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1))
        return {"t": round(float(t), 3), "df": round(float(df), 1), "p": round(float(p), 4),
                "d": round(float(d), 3), "n1": int(len(a)), "n2": int(len(b)),
                "g1": round(float(a.mean()), 3), "g2": round(float(b.mean()), 3),
                "g1sd": round(float(a.std(ddof=1)), 3), "g2sd": round(float(b.std(ddof=1)), 3)}

    def mwu(a, b):
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        r_rb = 1 - 2 * u / (n1 * n2)
        return {"U": round(float(u), 1), "p": round(float(p), 4), "r": round(float(abs(r_rb)), 3),
                "n1": int(n1), "n2": int(n2),
                "g1": round(float(a.mean()), 3), "g2": round(float(b.mean()), 3),
                "g1sd": round(float(a.std(ddof=1)), 3), "g2sd": round(float(b.std(ddof=1)), 3)}

    pv = data[data.private_vehicle_user == 1]
    nv = data[data.private_vehicle_user == 0]
    hi = data[data.awareness_score >= data.awareness_score.median()]
    lo = data[data.awareness_score < data.awareness_score.median()]
    short = data[data.commute_distance_cat.isin(["<2 km", "2-5 km"])]
    longd = data[~data.commute_distance_cat.isin(["<2 km", "2-5 km"])]

    res["groups"] = {
        "pv_intention": welch(pv.intention_score.values, nv.intention_score.values),
        "pv_awareness": welch(pv.awareness_score.values, nv.awareness_score.values),
        "pv_acceptance": welch(pv.acceptance_score.values, nv.acceptance_score.values),
        "dist_intention": welch(short.intention_score.values, longd.intention_score.values),
        "dist_carshare": mwu(short.car_share_A.values, longd.car_share_A.values),
        "dist_co2saved": mwu(short.estimated_co2_saved.values, longd.estimated_co2_saved.values),
        "aware_intention": welch(hi.intention_score.values, lo.intention_score.values),
        "aware_acceptance": welch(hi.acceptance_score.values, lo.acceptance_score.values),
        "aware_sustshare": mwu(hi.sust_share_A.values, lo.sust_share_A.values),
    }

    # ---- logistic regression ----
    X = data[["awareness_score", "concern_score", "convenience_score",
              "cost_sensitivity", "time_sensitivity", "acceptance_score"]]
    X = sm.add_constant(X)
    y = data["low_emission_majority"]
    logit = sm.Logit(y, X).fit(disp=0)
    from sklearn.metrics import roc_auc_score
    pred = logit.predict(X).values
    res["logit"] = {
        "n1": int(y.sum()), "n0": int((1 - y).sum()),
        "llr_chi2": round(float(logit.llr), 3), "llr_p": round(float(logit.llr_pvalue), 5),
        "prsquared": round(float(logit.prsquared), 3),
        "auc": round(float(roc_auc_score(y, pred)), 3),
        "correct": round(float(((pred > .5).astype(int) == y).mean() * 100), 1),
        "params": {k: round(float(v), 3) for k, v in logit.params.items()},
        "se": {k: round(float(v), 3) for k, v in logit.bse.items()},
        "pvalues": {k: round(float(v), 4) for k, v in logit.pvalues.items()},
        "odds": {k: round(float(np.exp(v)), 3) for k, v in logit.params.items()},
        "ci_low": {k: round(float(np.exp(v)), 3) for k, v in logit.conf_int()[0].items()},
        "ci_high": {k: round(float(np.exp(v)), 3) for k, v in logit.conf_int()[1].items()},
    }

    # ---- baseline vs recommendation (trip level) ----
    A = pd.concat([data[f"stageA_{s}"] for s in SCEN_ORDER]).reset_index(drop=True)
    C = pd.concat([data[f"stageC_{s}"] for s in SCEN_ORDER]).reset_index(drop=True)
    R = pd.concat([data[f"rec_{s}"] for s in SCEN_ORDER]).reset_index(drop=True)
    eA = pd.concat([data[f"emission_A_{s}"] for s in SCEN_ORDER]).reset_index(drop=True)
    eC = pd.concat([data[f"emission_C_{s}"] for s in SCEN_ORDER]).reset_index(drop=True)
    eCar = pd.concat([data[f"emission_car_{s}"] for s in SCEN_ORDER]).reset_index(drop=True)
    res["mode_choice"] = {}
    for m in ["C", "B", "M", "X", "Y", "E", "P", "S"]:
        nA, nC = int((A == m).sum()), int((C == m).sum())
        if nA or nC:
            res["mode_choice"][MODE_LABEL[m]] = {
                "code": m, "A_n": nA, "A_pct": round(100 * nA / len(A), 1),
                "C_n": nC, "C_pct": round(100 * nC / len(C), 1),
                "change_pp": round(100 * (nC - nA) / len(A), 1)}
    res["mode_choice"]["Sustainable (any non-car)"] = {
        "A_n": int((A != "C").sum()), "A_pct": round(100 * (A != "C").mean(), 1),
        "C_n": int((C != "C").sum()), "C_pct": round(100 * (C != "C").mean(), 1),
        "change_pp": round(100 * ((C != "C").mean() - (A != "C").mean()), 1)}

    # recommendation acceptance
    differs = (R != A)
    followed = ((C == R) & differs)
    n_diff = int(differs.sum())
    n_follow = int(followed.sum())
    carA = (A == "C"); carC = (C == "C")
    moved_off_car = int((carA & ~carC).sum())
    moved_to_car = int((~carA & carC).sum())
    mcnemar = stats.binomtest(moved_off_car, moved_off_car + moved_to_car, 0.5)
    w_car = stats.wilcoxon(data["car_share_A"], data["car_share_C"])
    # travel-time and cost differences
    tA = pd.concat([data[f"stageA_{s}"].map(lambda m: SCENARIOS[s]["options"][m][0]) for s in SCEN_ORDER]).reset_index(drop=True)
    tC = pd.concat([data[f"stageC_{s}"].map(lambda m: SCENARIOS[s]["options"][m][0]) for s in SCEN_ORDER]).reset_index(drop=True)
    cA = pd.concat([data[f"stageA_{s}"].map(lambda m: SCENARIOS[s]["options"][m][1]) for s in SCEN_ORDER]).reset_index(drop=True)
    cC = pd.concat([data[f"stageC_{s}"].map(lambda m: SCENARIOS[s]["options"][m][1]) for s in SCEN_ORDER]).reset_index(drop=True)
    dt = (tC - tA).values; dc = (cC - cA).values
    res["baseline_vs_rec"] = {
        "n_decisions": int(len(A)),
        "car_share_A": round(float(data["car_share_A"].mean()), 3),
        "car_share_C": round(float(data["car_share_C"].mean()), 3),
        "sust_share_A": round(float(data["sust_share_A"].mean()), 3),
        "sust_share_C": round(float(data["sust_share_C"].mean()), 3),
        "wilcoxon_car_W": round(float(w_car.statistic), 1), "wilcoxon_car_p": round(float(w_car.pvalue), 5),
        "rec_differs": n_diff, "rec_followed": n_follow,
        "acceptance_rate": round(100 * n_follow / n_diff, 1),
        "moved_off_car": moved_off_car, "moved_to_car": moved_to_car,
        "mode_shift_rate": round(100 * moved_off_car / int(carA.sum()), 1),
        "baseline_car_decisions": int(carA.sum()),
        "respondents_shift": int((data["shift_count"] >= 1).sum()),
        "respondents_shift_pct": round(100 * float((data["shift_count"] >= 1).mean()), 1),
        "mcnemar_p": round(float(mcnemar.pvalue), 6),
        "mean_time_diff": round(float(dt.mean()), 3),
        "time_diff_p": round(float(stats.wilcoxon(dt).pvalue), 4),
        "mean_cost_diff": round(float(dc.mean()), 3),
        "cost_diff_p": round(float(stats.wilcoxon(dc).pvalue), 5),
        "rec_mode_counts": {MODE_LABEL[m]: int((R == m).sum()) for m in R.unique()},
    }

    # ---- CO2 ----
    diff = (eC - eA).values
    pt = stats.ttest_rel(eC.values, eA.values)
    wz = stats.wilcoxon(diff)
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
    res["co2"] = {
        "baseline_mean": round(float(eA.mean()), 1),
        "recommended_mean": round(float(eC.mean()), 1),
        "saving_mean": round(float(-diff.mean()), 1),
        "ci_low": round(float(-ci[1]), 1), "ci_high": round(float(-ci[0]), 1),
        "pct_reduction": round(100 * float(-diff.mean()) / float(eA.mean()), 1),
        "t": round(float(pt.statistic), 3), "t_p": round(float(pt.pvalue), 6),
        "dz": round(float(-diff.mean() / diff.std(ddof=1)), 3),
        "wilcoxon_p": round(float(wz.pvalue), 6),
        "sw_diff_p": round(float(stats.shapiro(diff).pvalue), 5),
        "all_car_ref": round(float(eCar.mean()), 1),
        "by_scenario": {s: {
            "distance": SCENARIOS[s]["distance"],
            "all_car": round(float(data[f"emission_car_{s}"].mean()), 1),
            "A": round(float(data[f"emission_A_{s}"].mean()), 1),
            "C": round(float(data[f"emission_C_{s}"].mean()), 1),
            "saving": round(float((data[f"emission_A_{s}"] - data[f"emission_C_{s}"]).mean()), 1),
            "pct": round(100 * float((data[f"emission_A_{s}"] - data[f"emission_C_{s}"]).mean())
                         / float(data[f"emission_A_{s}"].mean()), 1)} for s in SCEN_ORDER},
    }

    # ---- projection ----
    per_trip = float(-diff.mean())
    res["projection"] = {
        "per_trip_g": round(per_trip, 1),
        "trips_per_month": 40, "months": 9, "cohort": 1000,
        "kg_month": round(per_trip * 40 / 1000, 1),
        "kg_year": round(per_trip * 360 / 1000, 1),
        "tonnes_cohort_year": round(per_trip * 360 * 1000 / 1e6, 1),
        "half_kg_month": round(per_trip * 0.5 * 40 / 1000, 1),
        "half_kg_year": round(per_trip * 0.5 * 360 / 1000, 1),
        "half_tonnes": round(per_trip * 0.5 * 360 * 1000 / 1e6, 1),
        "quarter_kg_month": round(per_trip * 0.25 * 40 / 1000, 1),
        "quarter_kg_year": round(per_trip * 0.25 * 360 / 1000, 1),
        "quarter_tonnes": round(per_trip * 0.25 * 360 * 1000 / 1e6, 1),
        "mean_scenario_distance": round(np.mean([SCENARIOS[s]["distance"] for s in SCEN_ORDER]), 1),
    }

    # ---- hypothesis summary ----
    def holm(ps):
        order = np.argsort(ps); m = len(ps); adj = np.empty(m); prev = 0
        for rank, i in enumerate(order):
            val = min(1.0, (m - rank) * ps[i]); val = max(val, prev); prev = val; adj[i] = val
        return adj
    hyps = [
        ("H1", "Spearman rho", "rho = %.2f" % res["correlations"]["awareness_score|intention_score"]["rho"],
         res["correlations"]["awareness_score|intention_score"]["p"]),
        ("H2", "Spearman rho", "rho = %.2f" % res["correlations"]["concern_score|shift_count"]["rho"],
         res["correlations"]["concern_score|shift_count"]["p"]),
        ("H3", "Spearman rho", "rho = %.2f" % res["correlations"]["convenience_score|acceptance_score"]["rho"],
         res["correlations"]["convenience_score|acceptance_score"]["p"]),
        ("H4", "Welch t", "t = %.2f" % res["groups"]["pv_intention"]["t"], res["groups"]["pv_intention"]["p"]),
        ("H5", "Logistic LR", "chi2(6) = %.2f" % res["logit"]["llr_chi2"], res["logit"]["llr_p"]),
        ("H6a", "Wilcoxon", "W = %.1f" % res["baseline_vs_rec"]["wilcoxon_car_W"],
         res["baseline_vs_rec"]["wilcoxon_car_p"]),
        ("H6b", "Paired t", "t = %.2f" % res["co2"]["t"], res["co2"]["t_p"]),
    ]
    adj = holm([h[3] for h in hyps])
    res["hypotheses"] = [{"id": h[0], "test": h[1], "result": h[2], "p": round(h[3], 4),
                          "holm_p": round(float(a), 4), "supported": bool(h[3] < 0.05)}
                         for h, a in zip(hyps, adj)]

    # ---- categorical descriptives ----
    def cat(col):
        vc = data[col].value_counts()
        order = sorted(vc.index) if col != "current_mode" else vc.index
        return [{"category": str(k), "n": int(vc[k]), "pct": round(100 * vc[k] / len(data), 1)}
                for k in order]
    res["profile"] = {c: cat(c) for c in
                      ["gender", "year_of_study", "programme", "commute_distance_cat",
                       "current_mode", "commute_days"]}
    res["profile_cont"] = {
        "age_mean": round(float(data.age.mean()), 2), "age_sd": round(float(data.age.std(ddof=1)), 2),
        "age_min": int(data.age.min()), "age_max": int(data.age.max()),
        "intention_midpoint_p": round(float(stats.wilcoxon(data.intention_score - 3).pvalue), 4),
        "awareness_high_pct": round(100 * float((data.awareness_score >= 4).mean()), 1),
        "low_emission_majority_pct": round(100 * float(data.low_emission_majority.mean()), 1)}
    # scenario-level option/choice summary for Appendix B verification
    res["scenario_choices"] = {}
    for s in SCEN_ORDER:
        vc = data[f"stageA_{s}"].value_counts()
        res["scenario_choices"][s] = {MODE_LABEL[k]: int(v) for k, v in vc.items()}

    # ---- item-level descriptives ----
    def item_stats(cols, labels):
        rows = []
        for c, lab in zip(cols, labels):
            x = data[c]
            rows.append({"item": c, "label": lab, "mean": round(float(x.mean()), 2),
                         "sd": round(float(x.std(ddof=1)), 2),
                         "agree_pct": round(100 * float((x >= 4).mean()), 1)})
        return rows
    res["items"] = {
        "awareness": item_stats([f"aw{i}" for i in range(1, 10)],
                                ["Knows which options emit less CO2 than a car",
                                 "Understands car commuting contributes to emissions",
                                 "Aware public transport can emit less per passenger-km",
                                 "Knows walking/cycling have almost no tailpipe emissions",
                                 "Aware carpooling reduces vehicles on the road",
                                 "Aware shared e-bikes/e-scooters can replace short car trips",
                                 "Understands EV benefit depends on electricity source",
                                 "Knows apps showing multimodal routes/emissions",
                                 "Can judge which option has lower CO2"]),
        "concern": item_stats([f"ec{i}" for i in range(1, 8)],
                              ["Concerned about road-traffic air pollution",
                               "Concerned about transport emissions and climate",
                               "Feels personal responsibility for travel emissions",
                               "Uncomfortable choosing high-emission option when cleaner exists",
                               "Believes individual choices make a difference",
                               "Universities should encourage low-emission commuting",
                               "Emission reduction important even if slightly slower"]),
        "intention": item_stats([f"int{i}" for i in range(1, 8)],
                                ["Walking", "Cycling", "Public transport", "Carpooling",
                                 "Shared mobility", "E-bike/e-scooter", "Electric vehicle"]),
    }

    # ---- conditional (multinomial) logit on Stage A choices ----
    rows = []
    for s in SCEN_ORDER:
        d = SCENARIOS[s]["distance"]
        for i in range(N):
            chosen = data.loc[i, f"stageA_{s}"]
            for m, (tm, co, com, cv) in SCENARIOS[s]["options"].items():
                rows.append({"scen": s, "rid": i, "mode": m,
                             "time": tm, "cost": co, "comfort": com, "conv": cv,
                             "emis": emission(m, d), "chosen": int(m == chosen)})
    cl = pd.DataFrame(rows)
    for v in ["time", "cost", "comfort", "conv", "emis"]:
        cl[v + "_z"] = (cl[v] - cl[v].mean()) / cl[v].std()
    cl["case"] = cl["scen"] + "_" + cl["rid"].astype(str)
    try:
        from statsmodels.discrete.conditional_models import ConditionalLogit
        mcl = ConditionalLogit(cl["chosen"], cl[["time_z", "cost_z", "comfort_z", "conv_z", "emis_z"]],
                               groups=cl["case"]).fit(disp=0)
        res["condlogit"] = {
            "params": {k: round(float(v), 3) for k, v in mcl.params.items()},
            "se": {k: round(float(v), 3) for k, v in mcl.bse.items()},
            "pvalues": {k: round(float(v), 4) for k, v in mcl.pvalues.items()},
            "llf": round(float(mcl.llf), 1)}
    except Exception as e:  # pragma: no cover
        res["condlogit"] = {"error": str(e)}

    # ---- Kruskal-Wallis across current mode groups ----
    groups = [g["intention_score"].values for _, g in data.groupby("current_mode")]
    kw = stats.kruskal(*groups)
    res["kruskal"] = {"H": round(float(kw.statistic), 3), "p": round(float(kw.pvalue), 4),
                      "groups": data.current_mode.nunique()}

    print(json.dumps({k: res[k] for k in ["n", "reliability"]}, indent=1))
    data.to_csv("simulated_dataset.csv", index=False)
    with open("analysis_output.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
