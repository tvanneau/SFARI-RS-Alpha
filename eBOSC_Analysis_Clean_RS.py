# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:16:30 2025

@author: tvanneau
"""

import pandas as pd
import numpy as np
import os, glob

ALPHA_MIN, ALPHA_MAX = 7.0, 13.0

# ---- Configure your roots and fixed total durations here ----
ROOTS = {
    "Resting_state": {
        "TD":  r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state\Extraction_eBOSC_V2\TD",
        "ASD": r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state\Extraction_eBOSC_V2\ASD",
        "SIB": r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state\Extraction_eBOSC_V2\SIB",
}}

TOTAL_DURATION_MAP = {
    "Resting_state": 50}

def compute_ibi_stats(group: pd.DataFrame) -> pd.Series:
    """
    Compute inter-burst interval (IBI) stats for one Subject × Paradigm × Channel group.
    IBI_i = Onset_i - Offset_{i-1}, after sorting by Onset.
    Returns mean, sd, and count of intervals.
    """
    g = group.sort_values("Onset").copy()

    # Previous burst offset
    prev_offset = g["Offset"].shift(1)

    # Interval from previous offset to current onset
    ibi = g["Onset"] - prev_offset

    # First burst has no previous offset => NaN; drop it
    ibi = ibi.dropna()

    # Optional: remove negative intervals (overlaps). Choose ONE policy:
    # Policy A (recommended): set overlaps to 0
    ibi = ibi.clip(lower=0)

    # Policy B: drop overlaps entirely
    # ibi = ibi[ibi >= 0]

    return pd.Series({
        "MeanIBI": ibi.mean() if len(ibi) else np.nan,
        "SDIBI": ibi.std(ddof=1) if len(ibi) > 1 else (0.0 if len(ibi) == 1 else np.nan),
        "NumIntervals": int(len(ibi))
    })

def _safe_concat(frames, **kwargs):
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, **kwargs) if frames else pd.DataFrame()

def compute_subject_data_fixed(folder_path: str, group_label: str, paradigm: str,
                               total_duration_map: dict) -> pd.DataFrame:
    """
    Compute per-subject, per-channel alpha metrics with abundance defined as
    sum(Duration_alpha) / FIXED_TOTAL_DURATION for the paradigm.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*_episodes.csv"))
    if not csv_files:
        return pd.DataFrame()

    fixed_total = float(total_duration_map[paradigm])  # hard fail if missing so bugs are obvious
    all_alpha_rows = []
    abundance_rows = []

    for file in csv_files:
        df = pd.read_csv(file)
        if "Subject" not in df.columns or "Channel" not in df.columns:
            continue

        subject_id = str(df["Subject"].iloc[0])

        # Alpha-only rows
        alpha_df = df[(df["FrequencyMean"] >= ALPHA_MIN) & (df["FrequencyMean"] <= ALPHA_MAX)].copy()
        if alpha_df.empty:
            # No alpha episodes -> still return rows with zeros if you want to keep subject/channel slots
            # Here we skip; optionally implement a skeleton row per channel if needed.
            continue

        alpha_df["Subject"]  = subject_id
        alpha_df["Group"]    = group_label
        all_alpha_rows.append(alpha_df)

        # Abundance per channel = sum(Duration_alpha)/fixed_total
        ch_abund = (alpha_df.groupby("Channel", as_index=False)["Duration"].sum()
                    .rename(columns={"Duration": "AlphaDuration"}))
        ch_abund["Abundance"] = ch_abund["AlphaDuration"] / fixed_total
        ch_abund["Abundance"] = ch_abund["Abundance"].clip(lower=0)
        ch_abund["Subject"]   = subject_id
        ch_abund["Group"]     = group_label
        abundance_rows.append(ch_abund[["Subject","Group","Channel","Abundance"]])

    alpha_data = _safe_concat(all_alpha_rows, ignore_index=True)
    abundance_df = _safe_concat(abundance_rows, ignore_index=True)

    if alpha_data.empty:
        return pd.DataFrame()

    # Per-subject, per-channel averages of alpha-episode properties
    # Per-subject, per-channel averages + variability (SD) of alpha-episode properties
    avg_df = (
        alpha_data.groupby(["Subject", "Group", "Channel"], as_index=False)
                 .agg(
                     AvgFrequency=("FrequencyMean", "mean"),
                     SDFrequency=("FrequencyMean", "std"),   # variability across episodes
                     AvgPower=("PowerMean", "mean"),
                     SDPower=("PowerMean", "std"),
                     AvgDuration=("Duration", "mean"),
                     SDDuration=("Duration", "std"),
                     NumEpisodes=("FrequencyMean", "count"),
                 )
    )
    
    ibi_df = (
    alpha_data.groupby(["Subject", "Group", "Channel"], as_index=False)
             .apply(compute_ibi_stats)
             .reset_index(drop=True)
             )
    
    avg_df["CVFrequency"] = avg_df["SDFrequency"] / avg_df["AvgFrequency"]
    avg_df["CVPower"]     = avg_df["SDPower"]     / avg_df["AvgPower"]
    avg_df["CVDuration"]  = avg_df["SDDuration"]  / avg_df["AvgDuration"]

    out = avg_df.merge(abundance_df, on=["Subject","Group","Channel"], how="left")
    out = out.merge(ibi_df, on=["Subject","Group","Channel"], how="left")
    return out

def load_all_paradigms_fixed(roots: dict, total_duration_map: dict) -> pd.DataFrame:
    frames = []
    for paradigm, groups in roots.items():
        for group, path in groups.items():
            frames.append(compute_subject_data_fixed(path, group, paradigm, total_duration_map))
    return _safe_concat(frames, ignore_index=True)

all_metrics = load_all_paradigms_fixed(ROOTS, TOTAL_DURATION_MAP)

#%% Plot Topomap for each parameter of interest

import mne
import numpy as np
import matplotlib.pyplot as plt

# Montage / info
montage = mne.channels.make_standard_montage('biosemi64')
info = mne.create_info(ch_names=montage.ch_names, sfreq=512, ch_types='eeg')
info.set_montage(montage)

def get_metric_topo(df, group, metric, info, agg="mean"):
    """
    df: long df with columns ['Subject','Group','Channel', metric]
    group: 'TD'/'ASD'/'SIB'
    metric: column name (e.g., 'AvgFrequency')
    agg: 'mean' (across subjects) or 'median'
    Returns array in info.ch_names order.
    """
    gdf = df[df["Group"] == group].copy()

    # Aggregate per channel across subjects
    if agg == "mean":
        ch_stat = gdf.groupby("Channel")[metric].mean()
    elif agg == "median":
        ch_stat = gdf.groupby("Channel")[metric].median()
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    # Reorder to montage channel order
    vals = np.array([ch_stat.get(ch, np.nan) for ch in info.ch_names], dtype=float)
    return vals

def plot_topos_for_metric(df, metric, info, groups=("TD","ASD","SIB"),
                          title=None, cmap="jet", robust=True):
    """
    robust=True uses 5th/95th percentiles for vmin/vmax (avoids outliers dominating).
    """
    topo = {g: get_metric_topo(df, g, metric, info, agg="mean") for g in groups}

    # Common vmin/vmax across groups
    all_vals = np.concatenate([topo[g][~np.isnan(topo[g])] for g in groups])
    if all_vals.size == 0:
        print(f"[SKIP] {metric}: no values found")
        return

    if robust:
        vmin, vmax = np.nanpercentile(all_vals, [5, 95])
        # vmin, vmax = 9.5,10.2

        if np.isclose(vmin, vmax):
            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    else:
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    fig, axes = plt.subplots(1, len(groups), figsize=(3.6*len(groups), 3.6))

    # Ensure axes is iterable
    if len(groups) == 1:
        axes = [axes]

    im = None
    for ax, g in zip(axes, groups):
        im, _ = mne.viz.plot_topomap(
            topo[g], info, axes=ax, show=False, vlim=(vmin, vmax), cmap=cmap
        )
        ax.set_title(f"{g}")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label(metric)

    fig.text(
        0.5, 0.98,
        title if title else metric,
        ha="center", va="top",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

metrics = [
    "Abundance",
    "AvgFrequency", "SDFrequency", "CVFrequency",
    "AvgPower", "SDPower", "CVPower",
    "AvgDuration", "SDDuration", "CVDuration",
    "NumEpisodes",
    "MeanIBI", "SDIBI", "NumIntervals"
]

for m in metrics:
    if m in all_metrics.columns:
        plot_topos_for_metric(all_metrics, m, info, title=f"Alpha episode metrics – {m}", robust=True)
    else:
        print("Missing metric column:", m)
        
#%%

import numpy as np
import pandas as pd

# Your 6 clusters (fix naming if you want)
cluster_channels = {
    "Left_centroparietal":  ['P5','P3','P1','CP5','CP3','CP1'],
    "Right_centroparietal": ['P2','P4','P6','CP2','CP4','CP6'],
    "Right_prefrontal":     ['F3','F1','FC3','FC1'],
    "Left_prefrontal":      ['F2','F4','FC2','FC4'],
    "Left_occipital":       ['O1', 'PO7', 'PO3'],
    "Right_occipital":      ['O2', 'PO8', 'PO4'],
}

# channel -> (Region, Hemisphere)
chan2region = {}
for cl, chans in cluster_channels.items():
    hemi = "Left" if "Left" in cl else "Right"
    if "occipital" in cl.lower():
        region = "Occipital"
    elif "prefrontal" in cl.lower():
        region = "Frontal"
    else:
        region = "Centroparietal"
    for ch in chans:
        chan2region[ch] = (region, hemi)

df = all_metrics.copy()
df["Region"] = df["Channel"].map(lambda ch: chan2region.get(ch, (np.nan, np.nan))[0])
df["Hemisphere"] = df["Channel"].map(lambda ch: chan2region.get(ch, (np.nan, np.nan))[1])

# Keep only channels that are in your 6-cluster definition
df = df.dropna(subset=["Region","Hemisphere"]).copy()

#%% OR average for each cluster for correlation with behavioral

import numpy as np
import pandas as pd

# -----------------------
# 1) Define channel clusters
# -----------------------
# cluster_channels = {
#     "Left_centroparietal":  ['P5','P3','P1','CP5','CP3','CP1'],
#     "Right_centroparietal": ['P2','P4','P6','CP2','CP4','CP6'],
#     "Left_prefrontal":      ['F3','F1','FC3','FC1'],   # <-- check naming: these are left-ish in your earlier code
#     "Right_prefrontal":     ['F2','F4','FC2','FC4'],   # <-- check naming: these are right-ish in your earlier code
#     "Left_occipital":       ['O1','PO7','PO3'],
#     "Right_occipital":      ['O2','PO8','PO4'],
# }

cluster_channels = {
    "Centroparietal":  ['P5','P3','P1','CP5','CP3','CP1','P2','P4','P6','CP2','CP4','CP6'],
    "Prefrontal":      ['F3','F1','FC3','FC1','F2','F4','FC2','FC4'],   # <-- check naming: these are left-ish in your earlier code
    "Occipital":       ['O1','PO7','PO3','O2','PO8','PO4'],
}

# If your naming is reversed (as in some earlier snippets), swap the two prefrontal lists.

# -----------------------
# 2) Build channel -> cluster mapping
# -----------------------
chan2cluster = {}
for cl, chans in cluster_channels.items():
    for ch in chans:
        chan2cluster[ch] = cl

# -----------------------
# 3) Helper: ROI-average function
# -----------------------
def make_cluster_averaged_df(
    df_all: pd.DataFrame,
    metrics: list,
    subject_col: str = "Subject",
    group_col: str = "Group",
    channel_col: str = "Channel",
    keep_cluster_cols: bool = True
) -> pd.DataFrame:
    """
    Convert per-channel df -> per-subject-per-cluster df by averaging metrics across channels in each cluster.
    """

    df = df_all.copy()

    # Clean strings
    df[channel_col] = df[channel_col].astype(str).str.strip()
    df[subject_col] = df[subject_col].astype(str).str.strip()
    df[group_col] = df[group_col].astype(str).str.strip()

    # Assign cluster
    df["Cluster"] = df[channel_col].map(chan2cluster)

    # Keep only channels that map to your clusters
    df = df.dropna(subset=["Cluster"]).copy()

    # Average within Subject x Group x Cluster
    out = (
        df.groupby([subject_col, group_col, "Cluster"], as_index=False)[metrics]
          .mean()
    )

    # Optional: add Region + Hemisphere columns from Cluster name
    if keep_cluster_cols:
        out["Hemisphere"] = out["Cluster"].str.extract(r"(Left|Right)", expand=False)
        out["Region"] = np.where(out["Cluster"].str.contains("occipital", case=False), "Occipital",
                        np.where(out["Cluster"].str.contains("prefrontal", case=False), "Frontal",
                                 "Centroparietal"))

    return out

# -----------------------
# 4) Choose which EEG metrics to average
# -----------------------
metrics = [
    "AvgFrequency",
    "SDFrequency",
    "AvgPower",
    "SDPower",
    "AvgDuration",
    "SDDuration",
    "NumEpisodes",
    "CVFrequency",
    "CVPower",
    "CVDuration",
    "Abundance",
    "MeanIBI",
    "SDIBI",
    "NumIntervals"
]

# Keep only metrics that exist in df_all (prevents KeyError)
metrics = [m for m in metrics if m in all_metrics.columns]

# -----------------------
# 5) Run
# -----------------------
df_cluster = make_cluster_averaged_df(
    df_all=all_metrics,
    metrics=metrics,
    subject_col="Subject",
    group_col="Group",      # change to "Group_x" if that's your column
    channel_col="Channel"
)

print(df_cluster.head())
print(df_cluster.groupby(["Group","Cluster"])["Subject"].nunique().head(20))

#%%

import numpy as np
import matplotlib.pyplot as plt

cluster_name = "Prefrontal"

df_lcp = df_cluster[df_cluster["Cluster"] == cluster_name].copy()

# Ensure consistent group order
group_order = ["ASD", "SIB", "TD"]
df_lcp["Group"] = df_lcp["Group"].astype(str).str.strip()
df_lcp["Group"] = pd.Categorical(df_lcp["Group"], categories=group_order, ordered=True)

# Quick check: one row per subject in this cluster
print("N subjects per group (Left_centroparietal):")
print(df_lcp.groupby("Group")["Subject"].nunique())

# If needed: drop duplicates defensively (shouldn’t be necessary if df_cluster is correct)
df_lcp = df_lcp.drop_duplicates(subset=["Subject", "Group", "Cluster"])

def box_jitter(ax, data, ycol, title, ylabel=None):
    # collect arrays by group
    groups = [g for g in data["Group"].cat.categories if g in data["Group"].unique()]
    arrays = [data.loc[data["Group"] == g, ycol].dropna().values for g in groups]

    # boxplot
    ax.boxplot(arrays, labels=groups, showfliers=False)

    # jittered scatter
    rng = np.random.default_rng(0)
    for i, arr in enumerate(arrays, start=1):
        x = rng.normal(loc=i, scale=0.06, size=len(arr))
        ax.scatter(x, arr, s=25, alpha=0.7)

    ax.set_title(title)
    ax.set_ylabel(ylabel if ylabel else ycol)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

box_jitter(
    axes[0],
    df_lcp,
    ycol="AvgFrequency",
    title="Alpha power (Centroparietal)",
    ylabel="Amplitude"
)

box_jitter(
    axes[1],
    df_lcp,
    ycol="Abundance",
    title="Alpha Abundance (Centroparietal)",
    ylabel="Abundance"
)

plt.show()

#%%

def violin_jitter(ax, data, ycol, title, ylabel=None):
    groups = [g for g in data["Group"].cat.categories if g in data["Group"].unique()]
    arrays = [data.loc[data["Group"] == g, ycol].dropna().values for g in groups]

    parts = ax.violinplot(arrays, showmeans=False, showmedians=True, showextrema=False)

    rng = np.random.default_rng(0)
    for i, arr in enumerate(arrays, start=1):
        x = rng.normal(loc=i, scale=0.06, size=len(arr))
        ax.scatter(x, arr, s=18, alpha=0.6)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel if ylabel else ycol)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

violin_jitter(
    axes[0], df_lcp, "AvgFrequency",
    "Alpha AvgFrequency distribution (Left centroparietal)",
    "AvgFrequency"
)

violin_jitter(
    axes[1], df_lcp, "Abundance",
    "Alpha burst duration distribution (Left centroparietal)",
    "AvgDuration"
)

plt.show()



#%%

# Endophenotype trend: ASD=-1, SIB=0, TD=+1
df["G_trend"] = df["Group"].map({"ASD": -1, "SIB": 0, "TD": 1}).astype(float)

# Deviation of SIB from midpoint of TD/ASD (optional; orthogonal to trend)
# ASD=-0.5, SIB=+1, TD=-0.5  (tests "SIB special" beyond linear trend)
df["G_sibdev"] = df["Group"].map({"ASD": -0.5, "SIB": 1.0, "TD": -0.5}).astype(float)


#%% Load and plot topomaps

import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# Folder where the CSVs live
data_dir = r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state_only\Stats_final\Stats_eBOSC_Single_Channels_Slope_Clinical.csv"

df = pd.read_csv(data_dir)

df['AvgPower'] = df['AvgPower']*1e-3
df['SDPower'] = np.log10(df['SDPower'])

a = df.columns

#%%

import statsmodels.formula.api as smf

# Choose your outcome metric here:
metric = "AvgFrequency"
dfm = df.dropna(subset=[metric, "G_trend", "Subject","Age","Sex"]).copy()

# Mixed model: subject random intercept
# Fixed: planned trend + region + hemisphere (+ interactions if you want)
md = smf.mixedlm(
    f"{metric} ~ G_trend + Age + Region + Hemisphere + G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)
m = md.fit(method="lbfgs")
print(m.summary())

#%% To include sex and age

import statsmodels.formula.api as smf

metric = "Abundance"
dfm = df.dropna(subset=[metric, "G_trend", "Subject", "Age", "Sex"]).copy()
dfm["Sex"] = dfm["Sex"].astype("category")

dfm["Sex"] = dfm["Sex"].astype(str).str.strip()
dfm["Sex"] = pd.Categorical(dfm["Sex"], categories=["Male", "Female"])

md = smf.mixedlm(
    f"{metric} ~ G_trend * Sex + Age + Region + Hemisphere + "
    f"G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())
#%%



import statsmodels.formula.api as smf

metric = "AvgPower"
dfm = df.dropna(subset=[metric, "G_trend", "Subject", "Age"]).copy()

md = smf.mixedlm(
    f"{metric} ~ G_trend + Age + Region + Hemisphere + G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)
m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "NumEpisodes"
dfm = df.dropna(subset=[metric, "G_trend", "Subject", "Age","FSIQ"]).copy()

md = smf.mixedlm(
    f"{metric} ~ G_trend + Age + FSIQ + Region + Hemisphere + G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)
m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "SDFrequency"

dfm = df.dropna(subset=[metric, "Group", "Subject", "Age"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Age + Region + Hemisphere + C(Group):Region + C(Group):Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

# Choose your outcome metric here:
metric = "AvgFrequency"
dfm = df.dropna(subset=[metric, "G_trend", "Subject"]).copy()

# Mixed model: subject random intercept
# Fixed: planned trend + region + hemisphere (+ interactions if you want)
md = smf.mixedlm(
    f"{metric} ~ G_trend + Region + Hemisphere + G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)
m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "AvgFrequency"

dfm = df.dropna(subset=[metric, "Group", "Subject"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Region + Hemisphere + C(Group):Region + C(Group):Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())

#%%
import statsmodels.formula.api as smf

# Choose your outcome metric here:
metric = "SDFrequency"
dfm = df.dropna(subset=[metric, "G_trend", "Subject"]).copy()

# Mixed model: subject random intercept
# Fixed: planned trend + region + hemisphere (+ interactions if you want)
md = smf.mixedlm(
    f"{metric} ~ G_trend + Region + Hemisphere + G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)
m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "SDFrequency"

dfm = df.dropna(subset=[metric, "Group", "Subject"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Region + Hemisphere + C(Group):Region + C(Group):Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())

#%%
import statsmodels.formula.api as smf

# Choose your outcome metric here:
metric = "AvgPower"
dfm = df.dropna(subset=[metric, "G_trend", "Subject"]).copy()

# Mixed model: subject random intercept
# Fixed: planned trend + region + hemisphere (+ interactions if you want)
md = smf.mixedlm(
    f"{metric} ~ G_trend + Region + Hemisphere + G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)
m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "AvgPower"

dfm = df.dropna(subset=[metric, "Group", "Subject"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Region + Hemisphere + C(Group):Region + C(Group):Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())

#%%
import statsmodels.formula.api as smf

# Choose your outcome metric here:
metric = "AvgDuration"
dfm = df.dropna(subset=[metric, "G_trend", "Subject"]).copy()

# Mixed model: subject random intercept
# Fixed: planned trend + region + hemisphere (+ interactions if you want)
md = smf.mixedlm(
    f"{metric} ~ G_trend + Region + Hemisphere + G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)
m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "AvgDuration"

dfm = df.dropna(subset=[metric, "Group", "Subject"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Region + Hemisphere + C(Group):Region + C(Group):Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())

#%%
import statsmodels.formula.api as smf

# Choose your outcome metric here:
metric = "NumEpisodes"
dfm = df.dropna(subset=[metric, "G_trend", "Subject"]).copy()

# Mixed model: subject random intercept
# Fixed: planned trend + region + hemisphere (+ interactions if you want)
md = smf.mixedlm(
    f"{metric} ~ G_trend + Region + Hemisphere + G_trend:Region + G_trend:Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)
m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "NumEpisodes"

dfm = df.dropna(subset=[metric, "Group", "Subject"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Region + Hemisphere + C(Group):Region + C(Group):Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())



#%%

import statsmodels.formula.api as smf

metric = "AlphaPeakPW"

dfm = df.dropna(subset=[metric, "Group", "Subject"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Region + Hemisphere + C(Group):Region + C(Group):Hemisphere",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import patsy

# -----------------------------
# Inputs you already have:
#   dfm : the dataframe you used to fit the MixedLM (has Region, Hemisphere, Group, G_trend, Subject, RelAlpha)
#   m   : fitted MixedLM result (md.fit(...))
# -----------------------------

# Make sure these are the same categories used in the model fit
regions = ["Centroparietal", "Frontal", "Occipital"]
hemis   = ["Left", "Right"]
groups  = ["ASD", "SIB", "TD"]

# Your coding
g_trend_map = {"ASD": -1.0, "SIB": 0.0, "TD": 1.0}

# Helper: build fixed-effect design matrix for new data using the model's design_info
def build_exog_from_model(mixedlm_result, new_df):
    design_info = mixedlm_result.model.data.design_info
    exog = patsy.build_design_matrices([design_info], new_df, return_type="dataframe")[0]
    return exog

# Predict mean and 95% CI using fixed-effects covariance (delta method)
def predict_fe_mean_ci(mixedlm_result, exog_row):
    beta = mixedlm_result.fe_params.values
    cov  = mixedlm_result.cov_params()  # includes fixed effects (top-left)
    # ensure we use fixed-effect part only
    fe_names = mixedlm_result.model.exog_names
    cov_fe = cov.loc[fe_names, fe_names].values if hasattr(cov, "loc") else cov[:len(beta), :len(beta)]

    x = np.asarray(exog_row).reshape(1, -1)
    mu = float(x @ beta)
    se = float(np.sqrt(x @ cov_fe @ x.T))
    ci = 1.96 * se
    return mu, mu - ci, mu + ci


# Build prediction grid: Group × Region, averaging over hemispheres by predicting both and averaging
rows = []
for g in groups:
    for r in regions:
        # predict for both hemispheres then average
        preds = []
        exogs = []
        for h in hemis:
            tmp = pd.DataFrame([{
                "G_trend": g_trend_map[g],
                "Region": r,
                "Hemisphere": h,
            }])
            exog = build_exog_from_model(m, tmp)
            exogs.append(exog.values[0])
            preds.append(exog.values[0])

        # average the two design rows => average prediction over hemispheres
        xbar = np.mean(np.vstack(exogs), axis=0)

        mu, lo, hi = predict_fe_mean_ci(m, xbar)

        rows.append({
            "Group": g,
            "Region": r,
            "Pred": mu,
            "CI_low": lo,
            "CI_high": hi
        })

emm = pd.DataFrame(rows)

# -----------------------------
# Plot: EMMs (Group on x) with lines by Region
# -----------------------------
group_order = ["ASD", "SIB", "TD"]
region_order = ["Centroparietal", "Occipital", "Frontal"]  # or any order you prefer

# numeric x positions for nice lines
xpos = {g:i for i,g in enumerate(group_order)}

plt.figure(figsize=(8.2, 4.8))

for r in region_order:
    sub = emm[emm["Region"] == r].copy()
    sub["x"] = sub["Group"].map(xpos)

    # line through means
    plt.plot(sub["x"], sub["Pred"], marker="o", linewidth=2, label=r)

    # error bars
    yerr = np.vstack([sub["Pred"] - sub["CI_low"], sub["CI_high"] - sub["Pred"]])
    plt.errorbar(sub["x"], sub["Pred"], yerr=yerr, fmt="none", capsize=6, linewidth=2)

plt.xticks(range(len(group_order)), group_order)
plt.xlabel("")
plt.ylabel("Model-predicted RelAlpha (mean ± 95% CI)")
plt.title("Resting-state relative alpha: TD→SIB→ASD gradient (MixedLM EMMs)\nAveraged over hemispheres; lines = Region")
plt.legend(frameon=False, title="Region")
plt.tight_layout()
plt.show()

#%%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Choose metric
# -----------------------------
metric = "AvgFrequency"   # change to RelAlpha etc.

# -----------------------------
# Liability coding
# -----------------------------
trend_map = {"ASD": -1, "SIB": 0, "TD": 1}
df["G_trend"] = df["Group"].map(trend_map).astype(float)

# -----------------------------
# Average per subject × hemisphere
# -----------------------------
subj_hemi = (
    df.groupby(["Subject", "Hemisphere", "Group", "G_trend"])[metric]
      .mean()
      .reset_index()
)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6,5))

sns.regplot(
    data=subj_hemi[subj_hemi["Hemisphere"] == "Left"],
    x="G_trend", y=metric,
    label="Left",
    scatter_kws=dict(s=40, alpha=0.6),
    line_kws=dict(lw=3)
)

sns.regplot(
    data=subj_hemi[subj_hemi["Hemisphere"] == "Right"],
    x="G_trend", y=metric,
    label="Right",
    scatter_kws=dict(s=40, alpha=0.6),
    line_kws=dict(lw=3)
)

# Pretty x labels
plt.xticks([-1, 0, 1], ["ASD", "SIB", "TD"])

plt.xlabel("Genetic liability gradient")
plt.ylabel(metric)
plt.title(f"{metric} – liability slope differs by hemisphere")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()


#%% Boxplot for a cluster of channel of interest

import matplotlib.pyplot as plt
import seaborn as sns

# Define channels of interest
channels_of_interest = ['P5','P3','P1','CP5','CP3','CP1']

df_td = all_metrics[all_metrics['Group'] == 'TD']
df_asd = all_metrics[all_metrics['Group'] == 'ASD']

# Filter for the channels of interest
td_filtered = df_td[df_td['Channel'].isin(channels_of_interest)]
asd_filtered = df_asd[df_asd['Channel'].isin(channels_of_interest)]

# Average duration per subject across these channels
td_avg = td_filtered.groupby('Subject')['Abundance'].mean().reset_index()
td_avg['Group'] = 'TD'

asd_avg = asd_filtered.groupby('Subject')['Abundance'].mean().reset_index()
asd_avg['Group'] = 'ASD'

# Combine the data
combined_df = pd.concat([td_avg, asd_avg], ignore_index=True)

# Plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=combined_df, x='Group', y='Abundance', palette='Set2')
sns.swarmplot(data=combined_df, x='Group', y='Abundance', color='black', alpha=0.7)
plt.title('Average Alpha Episode Duration\n(7–13 Hz) over Parieto-Occipital Channels')
plt.ylabel('Mean Duration (s)')
plt.xlabel('Group')
plt.tight_layout()
plt.show()

#%%

from scipy.stats import ttest_ind

# Extract data per group
td_values = combined_df[combined_df['Group'] == 'TD']['Abundance'].values
asd_values = combined_df[combined_df['Group'] == 'ASD']['Abundance'].values

# Run independent t-test
t_stat, p_val = ttest_ind(td_values, asd_values, equal_var=False)  # Welch's t-test

print(f"T-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")

#%%

# Need to install mne 1.6 to run the spatial permutation test 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne import create_info
from mne.channels import make_standard_montage
from mne.stats import spatio_temporal_cluster_test
from mne.viz import plot_topomap
from mne.channels import find_ch_adjacency
from scipy.stats import ttest_ind
from mne.stats import permutation_cluster_test
from scipy.stats import t

def stat_fun(*args):
    t_vals, _ = ttest_ind(*args, axis=0)
    return t_vals



# Step 1: Prepare Data
# Assuming 'Channel', 'Subject', and 'Abundance' columns in both dataframes
channels = make_standard_montage('biosemi64').ch_names

# Avg Duration
# td_data = df_td.pivot_table(index='Subject', columns='Channel', values='AvgDuration', fill_value=np.nan)
# asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='AvgDuration', fill_value=np.nan)

# AvgFrequency
# td_data = df_td.pivot_table(index='Subject', columns='Channel', values='AvgFrequency', fill_value=np.nan)
# asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='AvgFrequency', fill_value=np.nan)

# SDFrequency
# td_data = df_td.pivot_table(index='Subject', columns='Channel', values='SDFrequency', fill_value=np.nan)
# asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='SDFrequency', fill_value=np.nan)

# AvgPower
# td_data = df_td.pivot_table(index='Subject', columns='Channel', values='AvgPower', fill_value=np.nan)
# asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='AvgPower', fill_value=np.nan)

# NumEpisodes
# td_data = df_td.pivot_table(index='Subject', columns='Channel', values='NumEpisodes', fill_value=np.nan)
# asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='NumEpisodes', fill_value=np.nan)

# CVFrequency
# td_data = df_td.pivot_table(index='Subject', columns='Channel', values='CVFrequency', fill_value=np.nan)
# asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='CVFrequency', fill_value=np.nan)

# MeanIBI
td_data = df_td.pivot_table(index='Subject', columns='Channel', values='MeanIBI', fill_value=np.nan)
asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='MeanIBI', fill_value=np.nan)




# Function to fill NaN values with the mean value of the corresponding channel
def fill_nan_with_channel_mean(data):
    """
    Replace NaN values in each channel with the mean value of that channel.
    
    Parameters:
    - data (pd.DataFrame): DataFrame where rows are subjects and columns are channels.
    
    Returns:
    - pd.DataFrame: DataFrame with NaNs replaced by channel-wise means.
    """
    return data.apply(lambda col: col.fillna(col.mean()), axis=0)

# Ensure `td_data` and `asd_data` are defined before running this code
if 'td_data' in globals() and 'asd_data' in globals():
    # Apply the function to both TD and ASD data matrices
    td_data_filled = fill_nan_with_channel_mean(td_data.copy())
    asd_data_filled = fill_nan_with_channel_mean(asd_data.copy())

else:
    print("Error: Ensure that 'td_data' and 'asd_data' are defined before running this script.")

# Ensure the same channel order for both groups
td_data = td_data_filled[channels]
asd_data = asd_data_filled[channels]

# Convert dataframes to numpy arrays
td_values = td_data.to_numpy()
asd_values = asd_data.to_numpy()

# Step 2: Define the adjacency matrix
montage = make_standard_montage('biosemi64')
info = create_info(channels, sfreq=512, ch_types='eeg')
info.set_montage(montage)

# Create adjacency matrix based on the spatial layout of the electrodes
adjacency, ch_names = find_ch_adjacency(info, ch_type='eeg')
# mne.viz.plot_ch_adjacency(raw.info, adjacency, ch_names)
# Step 3: Spatial Cluster-Based Permutation Test
X = [td_values, asd_values]  # No need to stack, pass as a list

# Determine the cluster-forming threshold based on a critical t-value
n_participants = 40  # Assume equal sample size for simplicity
alpha_level = 0.05  # Desired significance level for the threshold
df = n_participants - 1  # Degrees of freedom for each group
t_threshold = abs(t.ppf(alpha_level / 2, df))  # Two-tailed t-value for alpha level

# Classical permutation test
# Perform the test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    X, stat_fun=stat_fun, n_permutations=1000, threshold=t_threshold, adjacency=adjacency,out_type = 'indices',
    tail=1, n_jobs=1)

# # Perform the test
# T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
#     X, stat_fun=None, n_permutations=1000, threshold=None, adjacency=adjacency,out_type = 'indices',
#     tail=1, n_jobs=1, seed=42
# )

#%%

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Define color limits
vlim = np.max(np.abs(T_obs))  

# Plot the T-values as a topomap and get the image handle
im, _ = plot_topomap(T_obs, info, cmap='RdBu_r', vlim=(-vlim, vlim), axes=ax, show=False)
ax.set_title('T-values (TD vs ASD)')

# Plot significant clusters
mask = np.zeros_like(T_obs, dtype=bool)
for i, cluster in enumerate(clusters):
    if cluster_p_values[i] < 0.05:  # Significance threshold
        significant_indices = cluster  # Extract indices
        mask[significant_indices] = True  # Apply mask to those indices

plot_topomap(T_obs, info, mask=mask, cmap='RdBu_r', vlim=(-vlim, vlim), axes=ax, show=True)

# Add a colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('T-values')

plt.show()

#%% On all_complete

import pandas as pd
from collections import defaultdict

# ------------ Configure ------------
metrics = [
    "NumEpisodes","AvgFrequency","AvgPower","AvgDuration","Abundance",
    "SDPower","SDFrequency","SDDuration","CVFrequency","CVPower","CVDuration",
    "MeanIBI","SDIBI"
]

cluster_channels = {
    "Left_centroparietal":  ['P5','P3','P1','CP5','CP3','CP1'],
    "Right_centroparietal": ['P2','P4','P6','CP2','CP4','CP6'],
    "Right_prefrontal":     ['F3','F1','FC3','FC1'],
    "Left_prefrontal":      ['F2','F4','FC2','FC4'],
    "Left_occipital":       ['O1','PO7','PO3'],
    "Right_occipital":      ['O2','PO8','PO4'],
}

# ------------ Helpers ------------
def _build_chan2clusters(cluster_channels):
    chan2clusters = defaultdict(list)
    for cl_name, chans in cluster_channels.items():
        for ch in chans:
            chan2clusters[ch].append(cl_name)
    return chan2clusters

def _add_cluster_column(df, chan2clusters):
    df = df.copy()
    df["Channel"] = df["Channel"].astype(str).str.strip()
    df["Cluster"] = df["Channel"].map(lambda ch: chan2clusters.get(ch, []))
    df = df.explode("Cluster", ignore_index=True)
    df = df[df["Cluster"].notna() & (df["Cluster"] != "")]
    return df

def make_cluster_level_rs(all_complete: pd.DataFrame, metrics, cluster_channels,
                          group_col="Group") -> pd.DataFrame:
    """
    Input: all_complete with columns: Subject, Group, Channel, metrics...
    Output: Subject × Group × Cluster with cluster-mean of metrics.
    """
    cols_needed = ["Subject", group_col, "Channel"] + metrics
    df = all_complete.loc[:, cols_needed].copy()

    # keep only channels used in any cluster
    chan2clusters = _build_chan2clusters(cluster_channels)
    keep_chans = set(chan2clusters.keys())
    df["Channel"] = df["Channel"].astype(str).str.strip()
    df = df[df["Channel"].isin(keep_chans)]

    # map channels -> cluster
    dfc = _add_cluster_column(df, chan2clusters)

    # average within Subject × Group × Cluster
    level = (dfc.groupby(["Subject", group_col, "Cluster"], as_index=False)
                .agg({m: "mean" for m in metrics}))

    return level

# ------------ Usage ------------
level_wide_rs = make_cluster_level_rs(all_metrics, metrics, cluster_channels, group_col="Group")
#%%
# Save results to a CSV file
level_wide_rs.to_csv("LMMs_eBOSC_RS_V2_2.csv", index=False)

