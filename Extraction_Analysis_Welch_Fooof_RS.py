# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 17:03:00 2026

@author: tvanneau
"""

#%%

import matplotlib as mpl

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

#%%

import mne
import numpy as np
import pandas as pd
import os
from fooof import FOOOF

file_path = r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state\Data_eBOSC\ASD"
save_dir  = r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state\Extraction_PSD_Fooof_linear"
os.makedirs(save_dir, exist_ok=True)

list_files_epochs = sorted([s for s in os.listdir(file_path) if "_EEG_epo" in s])

subjects_to_fix = {"11576"}

bands = {
    "theta": (3, 7),
    "alpha": (7, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 35),
    "total": (3, 35)
}

freqs = np.linspace(1, 35, num=200)

def bandpower_from_psd(psd, freqs, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapz(psd[mask], freqs[mask]))

def compute_bandpowers(psd, freqs, bands):
    abs_power = {name: bandpower_from_psd(psd, freqs, fmin, fmax)
                 for name, (fmin, fmax) in bands.items()}
    total = abs_power.get("total", np.nan)
    rel_power = {}
    for name in bands:
        if name == "total":
            continue
        rel_power[name] = abs_power[name] / total if np.isfinite(total) and total > 0 else np.nan
    return abs_power, rel_power

def find_alpha_peak(freqs, psd, fmin=7, fmax=13):
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() < 1:
        return np.nan, np.nan
    idx = np.argmax(psd[mask])
    return float(freqs[mask][idx]), float(psd[mask][idx])

def iaf_barycenter(freqs, psd, fmin=7.0, fmax=13.0):
    mask = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[mask]
    p = psd[mask]
    good = np.isfinite(f) & np.isfinite(p) & (p > 0)
    f, p = f[good], p[good]
    if p.size == 0:
        return np.nan
    denom = p.sum()
    return float((f * p).sum() / denom) if denom > 0 else np.nan

def get_fooof_aperiodic(fm):
    ap = fm.get_params("aperiodic_params")
    if len(ap) == 2:
        offset, exponent = ap
        # knee = np.nan
    else:
        offset, knee, exponent = ap
    return float(offset), float(exponent)

rows = []
psd_rows = []

for fname in list_files_epochs:
    fpath = os.path.join(file_path, fname)
    epochs = mne.read_epochs(fpath, preload=True)
    # Keep EEG only (drops EOG/ECG/EMG/EXG/Status/etc.)
    epochs = epochs.copy().pick_types(eeg=True, eog=False, ecg=False, emg=False, misc=False, stim=False)

    subject_id = fname.split("_")[0]

    if subject_id in subjects_to_fix:
        print(f"Dropping first epoch for subject {subject_id}")
        epochs.drop([0])

    epochs.set_eeg_reference("average")

    # Use FIRST epoch and crop 5–55s
    ep = epochs[0].copy().crop(tmin=5, tmax=55)

    data = ep.get_data().squeeze()  # (n_channels, n_times)
    if data.ndim == 1:
        data = data[None, :]

    ch_names = ep.ch_names
    sfreq = float(ep.info["sfreq"])
    n_fft = max(256, int(round(2 * sfreq)))  # ~2s window

    # Compute PSD for all channels at once (faster)
    psds, psd_freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=float(freqs.min()),
        fmax=float(freqs.max()),
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        average="mean",
        verbose=False
    )  # shape: (n_channels, n_freqs_native)

    for ci, ch in enumerate(ch_names):
        psd = psds[ci]
        psd_interp = np.interp(freqs, psd_freqs, psd)

        # save PSD (for later plotting)
        for f, p in zip(freqs, psd_interp):
            psd_rows.append({"Subject": subject_id, "Channel": ch, "Frequency": f, "PSD": p})

        # band powers
        abs_power, rel_power = compute_bandpowers(psd_interp, freqs, bands)

        # peaks
        paf, paf_amp = find_alpha_peak(freqs, psd_interp, 7, 13)
        iaf = iaf_barycenter(freqs, psd_interp, 7, 13)

        # FOOOF
        fm = FOOOF(aperiodic_mode="knee", verbose=False)
        fm.fit(freqs, psd_interp)

        # offset, exponent, knee = get_fooof_aperiodic(fm)
        
        offset, exponent = get_fooof_aperiodic(fm)


        # aperiodic-removed (flattened) barycenter IAF
        # internal ap fit is in log10 space
        aper_fit = fm._ap_fit
        log_psd = np.log10(psd_interp + 1e-20)
        flat_lin = 10 ** (log_psd - aper_fit)
        iaf_flat = iaf_barycenter(freqs, flat_lin, 7, 13)

        # alpha peaks from FOOOF
        peaks = fm.get_params("peak_params")
        alpha_cf = alpha_pw = alpha_bw = np.nan
        alpha_pw_sum = np.nan
        n_alpha_peaks = 0

        if peaks is not None and not (isinstance(peaks, np.ndarray) and peaks.size == 0):
            peaks = np.atleast_2d(peaks)
            alpha_peaks = peaks[(peaks[:, 0] >= 7) & (peaks[:, 0] <= 13)]
            n_alpha_peaks = int(alpha_peaks.shape[0])
            if n_alpha_peaks > 0:
                idx = np.nanargmax(alpha_peaks[:, 1])
                alpha_cf, alpha_pw, alpha_bw = alpha_peaks[idx]
                alpha_pw_sum = float(np.nansum(alpha_peaks[:, 1]))

        rows.append({
            "Subject": subject_id,
            "Group": "SIB",
            "Channel": ch,

            "AbsTheta": abs_power["theta"],
            "AbsAlpha": abs_power["alpha"],
            "AbsLowBeta": abs_power["low_beta"],
            "AbsHighBeta": abs_power["high_beta"],

            "RelTheta": rel_power["theta"],
            "RelAlpha": rel_power["alpha"],
            "RelLowBeta": rel_power["low_beta"],
            "RelHighBeta": rel_power["high_beta"],

            "PAF": paf,
            "PAF_amp": paf_amp,
            "IAF_bary": iaf,
            "IAF_flat_bary": iaf_flat,

            "AP_offset": offset,
            "AP_exponent": exponent,
            # "AP_knee": knee,

            "AlphaPeaksN": n_alpha_peaks,
            "AlphaPeakCF": alpha_cf,
            "AlphaPeakPW": alpha_pw,
            "AlphaPeakBW": alpha_bw,
            "AlphaPeakPWsum": alpha_pw_sum,
        })

df_chan = pd.DataFrame(rows)
df_psd  = pd.DataFrame(psd_rows)

#%%

df_chan.to_csv(os.path.join(save_dir, "Welch_FOOOF_perChannel_metrics_ASD.csv"), index=False)
df_psd.to_csv(os.path.join(save_dir, "Welch_PSD_perChannel_ASD.csv"), index=False)

#%% Load and plot topomaps

import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# Folder where the CSVs live
data_dir = r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state\Extraction_PSD_Fooof_linear"

# Filenames (adjust if yours differ)
files = {
    "TD":  os.path.join(data_dir, "Welch_FOOOF_perChannel_metrics_TD.csv"),
    "ASD": os.path.join(data_dir, "Welch_FOOOF_perChannel_metrics_ASD.csv"),
    "SIB": os.path.join(data_dir, "Welch_FOOOF_perChannel_metrics_SIB.csv"),
}

dfs = []
for g, f in files.items():
    df = pd.read_csv(f)
    # Ensure Group exists / correct
    df["Group"] = g
    # Clean channel names
    df["Channel"] = df["Channel"].astype(str).str.strip()
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

print(df_all.groupby("Group")["Subject"].nunique())
print("Example channels:", df_all["Channel"].unique()[:10])

df_all["AbsAlpha_log"] = np.log10(df_all["AbsAlpha"])
df_all["AbsTheta_log"] = np.log10(df_all["AbsTheta"])
df_all["AbsLowBeta_log"] = np.log10(df_all["AbsLowBeta"])
df_all["AbsHighBeta_log"] = np.log10(df_all["AbsHighBeta"])

montage = mne.channels.make_standard_montage("biosemi64")
info = mne.create_info(ch_names=montage.ch_names, sfreq=512, ch_types="eeg")
info.set_montage(montage)

# (Optional) Keep only channels that exist in the montage to avoid mismatches
df_all = df_all[df_all["Channel"].isin(info.ch_names)].copy()

def group_mean_topo(df, group, metric, info):
    gdf = df[df["Group"] == group]
    ch_mean = gdf.groupby("Channel")[metric].mean()
    return np.array([ch_mean.get(ch, np.nan) for ch in info.ch_names], dtype=float)

def shared_vlim(arrs, robust=True):
    vals = np.concatenate([a[np.isfinite(a)] for a in arrs])
    if vals.size == 0:
        return (np.nan, np.nan)
    if robust:
        vmin, vmax = np.nanpercentile(vals, [5, 95])
    else:
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if np.isclose(vmin, vmax):
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    return (float(vmin), float(vmax))

def plot_topomap_3groups(df, metric, info, title=None, robust=True, cmap=None):
    groups = ["TD", "ASD", "SIB"]
    topo = {g: group_mean_topo(df, g, metric, info) for g in groups}

    vmin, vmax = shared_vlim([topo[g] for g in groups], robust=robust)
    # vmin, vmax = 9.4,10.25
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        print(f"[SKIP] {metric}: no finite values.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    im = None

    for ax, g in zip(axes, groups):
        im, _ = mne.viz.plot_topomap(
            topo[g], info, axes=ax, show=False, vlim=(vmin, vmax), cmap=cmap
        )
        ax.set_title(g)

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

alpha_metrics = [
    "AbsAlpha", "RelAlpha",
    "PAF", "PAF_amp",
    "IAF_bary", "IAF_flat_bary",
    "AlphaPeaksN", "AlphaPeakCF", "AlphaPeakPW", "AlphaPeakBW", "AlphaPeakPWsum",    "AP_offset",
        "AP_exponent",
        "AP_knee","AbsAlpha_log"
]

for m in alpha_metrics:
    if m in df_all.columns:
        plot_topomap_3groups(df_all, m, info, title=f"Resting-state – {m}", robust=True, cmap= "jet")
    else:
        print("Missing column:", m)
        
#%% Keep single channel rows for LMMs later

import numpy as np
import pandas as pd

# Your 6 clusters (fix naming if you want)
cluster_channels = {
    "Left_centroparietal":  ['P5','P3','P1','CP5','CP3','CP1'],
    "Right_centroparietal": ['P2','P4','P6','CP2','CP4','CP6'],
    "Left_prefrontal":     ['F3','F1','FC3','FC1'],
    "Right_prefrontal":      ['F2','F4','FC2','FC4'],
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

df = df_all.copy()
df["Region"] = df["Channel"].map(lambda ch: chan2region.get(ch, (np.nan, np.nan))[0])
df["Hemisphere"] = df["Channel"].map(lambda ch: chan2region.get(ch, (np.nan, np.nan))[1])

# Keep only channels that are in your 6-cluster definition
df = df.dropna(subset=["Region","Hemisphere"]).copy()

#%%

df.to_csv("Welch_Fooof_PSD_Linear.csv", index=False)

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
    "AbsTheta",
    "AbsAlpha",
    "AbsLowBeta",
    "AbsHighBeta",
    "RelTheta",
    "RelAlpha",
    "RelLowBeta",
    "RelHighBeta",
    "PAF",
    "PAF_amp",
    "IAF_bary",
    "IAF_flat_bary",
    "AP_offset",
    "AP_exponent",
    "AP_knee",
    "AlphaPeaksN",
    "AlphaPeakCF",
    "AlphaPeakPW",
    "AlphaPeakPWsum" 
]

# Keep only metrics that exist in df_all (prevents KeyError)
metrics = [m for m in metrics if m in df_all.columns]

# -----------------------
# 5) Run
# -----------------------
df_cluster = make_cluster_averaged_df(
    df_all=df_all,
    metrics=metrics,
    subject_col="Subject",
    group_col="Group",      # change to "Group_x" if that's your column
    channel_col="Channel"
)

print(df_cluster.head())
print(df_cluster.groupby(["Group","Cluster"])["Subject"].nunique().head(20))

#%% Raincloud plots for one selected channel cluster

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# User settings
# ============================================================

output_dir = os.path.join(
    data_dir,
    "Raincloud_plots"
)
os.makedirs(output_dir, exist_ok=True)

groups = ["ASD", "SIB", "TD"]

group_colors = {
    "ASD":  "#377EB8",
    "SIB": "#E41A1C",
    "TD": "#4DAF4A"
}

# BioSemi-64 channel clusters
cluster_channels = {
    "Left_centroparietal": [
        "P5", "P3", "P1",
        "CP5", "CP3", "CP1"
    ],
    "Right_centroparietal": [
        "P2", "P4", "P6",
        "CP2", "CP4", "CP6"
    ],
    "Left_prefrontal": [
        "F3", "F1",
        "FC3", "FC1"
    ],
    "Right_prefrontal": [
        "F2", "F4",
        "FC2", "FC4"
    ],
    "Left_occipital": [
        "O1", "PO7", "PO3"
    ],
    "Right_occipital": [
        "O2", "PO8", "PO4"
    ]
}

# Select the cluster to plot
selected_cluster = "Left_centroparietal"

# Parameters for which one figure will be produced
parameters_to_plot = [
    "AbsAlpha",
    "RelAlpha",
    "PAF",
    "PAF_amp",
    "IAF_bary",
    "IAF_flat_bary",
    "AlphaPeaksN",
    "AlphaPeakCF",
    "AlphaPeakPW",
    "AlphaPeakBW",
    "AlphaPeakPWsum",
    "AP_offset",
    "AP_exponent",
    "AP_knee",
    "AbsAlpha_log"
]

# Optional display labels
parameter_labels = {
    "AbsAlpha": "Absolute alpha power",
    "RelAlpha": "Relative alpha power",
    "PAF": "Peak alpha frequency (Hz)",
    "PAF_amp": "Peak alpha amplitude",
    "IAF_bary": "Individual alpha frequency (Hz)",
    "IAF_flat_bary": "Aperiodic-adjusted IAF (Hz)",
    "AlphaPeaksN": "Number of alpha peaks",
    "AlphaPeakCF": "Alpha peak frequency (Hz)",
    "AlphaPeakPW": "Alpha peak power",
    "AlphaPeakBW": "Alpha peak bandwidth (Hz)",
    "AlphaPeakPWsum": "Summed alpha peak power",
    "AP_offset": "Aperiodic offset",
    "AP_exponent": "Aperiodic exponent",
    "AP_knee": "Aperiodic knee",
    "AbsAlpha_log": r"Absolute alpha power [log$_{10}$]"
}

save_figures = False

# ============================================================
# Validate selected cluster
# ============================================================

if selected_cluster not in cluster_channels:
    raise ValueError(
        f"Unknown cluster: {selected_cluster}\n"
        f"Available clusters: {list(cluster_channels.keys())}"
    )

selected_channels = cluster_channels[selected_cluster]

available_channels = sorted(
    set(selected_channels).intersection(df_all["Channel"].unique())
)

missing_channels = sorted(
    set(selected_channels).difference(df_all["Channel"].unique())
)

print(f"Selected cluster: {selected_cluster}")
print(f"Channels used: {available_channels}")

if missing_channels:
    print(f"Missing channels: {missing_channels}")

if len(available_channels) == 0:
    raise ValueError(
        "None of the selected channels were found in df_all."
    )

# ============================================================
# Function: participant-level cluster average
# ============================================================

def calculate_participant_cluster_values(
    dataframe,
    metric,
    channels
):
    """
    Calculate one cluster value per participant.

    The function first averages duplicate rows within each
    participant/channel and then averages channels within each
    participant.
    """

    required_columns = [
        "Subject",
        "Group",
        "Channel",
        metric
    ]

    missing_columns = [
        col for col in required_columns
        if col not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for {metric}: {missing_columns}"
        )

    cluster_df = dataframe[
        dataframe["Channel"].isin(channels)
    ].copy()

    cluster_df[metric] = pd.to_numeric(
        cluster_df[metric],
        errors="coerce"
    )

    # Remove infinite values
    cluster_df[metric] = cluster_df[metric].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # One value per participant and channel
    subject_channel = (
        cluster_df
        .groupby(
            ["Subject", "Group", "Channel"],
            as_index=False
        )
        .agg(
            channel_value=(metric, "mean")
        )
    )

    # One cluster average per participant
    subject_cluster = (
        subject_channel
        .groupby(
            ["Subject", "Group"],
            as_index=False
        )
        .agg(
            value=("channel_value", "mean"),
            n_channels=("channel_value", "count")
        )
    )

    # Number of possible channels in this cluster
    subject_cluster["n_possible_channels"] = len(channels)

    subject_cluster["proportion_channels"] = (
        subject_cluster["n_channels"]
        / subject_cluster["n_possible_channels"]
    )

    subject_cluster["parameter"] = metric
    subject_cluster["cluster"] = selected_cluster

    return subject_cluster


# ============================================================
# Function: raincloud plot
# ============================================================

def plot_raincloud(
    participant_data,
    metric,
    ylabel=None,
    title=None,
    save=False
):
    """
    Plot half-violins, boxplots, and individual participant points.
    """

    plot_df = participant_data.dropna(
        subset=["value"]
    ).copy()

    plot_df = plot_df[
        plot_df["Group"].isin(groups)
    ].copy()

    if plot_df.empty:
        print(f"[SKIP] {metric}: no valid participant values")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    positions = np.arange(len(groups))
    random_generator = np.random.default_rng(42)

    for position, group in zip(positions, groups):

        group_values = (
            plot_df.loc[
                plot_df["Group"] == group,
                "value"
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(group_values) == 0:
            continue

        color = group_colors[group]

        # ----------------------------------------------------
        # Half violin: distribution shown on the left
        # ----------------------------------------------------

        if (
            len(group_values) >= 2
            and np.nanstd(group_values) > 0
        ):
            violin = ax.violinplot(
                group_values,
                positions=[position],
                widths=0.75,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method="scott"
            )

            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.35)
                body.set_linewidth(1.2)

                # Keep only the left half of the violin
                for path in body.get_paths():
                    vertices = path.vertices
                    vertices[:, 0] = np.minimum(
                        vertices[:, 0],
                        position
                    )

        # ----------------------------------------------------
        # Narrow boxplot
        # ----------------------------------------------------

        box_position = position + 0.08

        box = ax.boxplot(
            group_values,
            positions=[box_position],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            zorder=3
        )

        box["boxes"][0].set(
            facecolor="white",
            edgecolor=color,
            linewidth=1.5
        )

        box["medians"][0].set(
            color="black",
            linewidth=2
        )

        for whisker in box["whiskers"]:
            whisker.set(
                color=color,
                linewidth=1.3
            )

        for cap in box["caps"]:
            cap.set(
                color=color,
                linewidth=1.3
            )

        # ----------------------------------------------------
        # Individual participant points: "rain"
        # ----------------------------------------------------

        jitter = random_generator.uniform(
            low=0.17,
            high=0.31,
            size=len(group_values)
        )

        point_positions = position + jitter

        ax.scatter(
            point_positions,
            group_values,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
            zorder=4
        )

        # Sample size above each group
        group_min = np.nanmin(group_values)
        group_max = np.nanmax(group_values)
        group_range = group_max - group_min

        if group_range == 0:
            group_range = abs(group_max) * 0.05

        if group_range == 0:
            group_range = 1.0

        ax.text(
            position,
            group_max + 0.07 * group_range,
            f"n={len(group_values)}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=12)

    ax.set_xlim(-0.55, len(groups) - 0.50)

    ax.set_xlabel("Group", fontsize=12)

    if ylabel is None:
        ylabel = metric

    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f"{selected_cluster}: {metric}"

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    plt.tight_layout()

    if save:
        safe_cluster_name = selected_cluster.replace(" ", "_")
        safe_metric_name = metric.replace(" ", "_")

        png_file = os.path.join(
            output_dir,
            f"Raincloud_{safe_cluster_name}_{safe_metric_name}.png"
        )

        svg_file = os.path.join(
            output_dir,
            f"Raincloud_{safe_cluster_name}_{safe_metric_name}.svg"
        )

        fig.savefig(
            png_file,
            dpi=300,
            bbox_inches="tight"
        )

        fig.savefig(
            svg_file,
            bbox_inches="tight"
        )

        print(f"Saved: {png_file}")

    plt.show()
    # plt.close(fig)


# ============================================================
# Calculate and plot every parameter
# ============================================================

all_participant_cluster_values = []

for metric in parameters_to_plot:

    if metric not in df_all.columns:
        print(f"[SKIP] Missing parameter: {metric}")
        continue

    participant_values = calculate_participant_cluster_values(
        dataframe=df_all,
        metric=metric,
        channels=available_channels
    )

    all_participant_cluster_values.append(
        participant_values
    )

    n_valid = (
        participant_values
        .dropna(subset=["value"])
        .groupby("Group")["Subject"]
        .nunique()
    )

    print(f"\n{selected_cluster} — {metric}")
    print(n_valid)

    plot_raincloud(
        participant_data=participant_values,
        metric=metric,
        ylabel=parameter_labels.get(metric, metric),
        title=(
            f"{selected_cluster.replace('_', ' ')} — "
            f"{parameter_labels.get(metric, metric)}"
        ),
        save=save_figures
    )


# ============================================================
# Save participant-level cluster values
# ============================================================

# if len(all_participant_cluster_values) > 0:

#     df_cluster_values = pd.concat(
#         all_participant_cluster_values,
#         ignore_index=True
#     )

#     csv_file = os.path.join(
#         output_dir,
#         f"{selected_cluster}_participant_level_metrics.csv"
#     )

#     df_cluster_values.to_csv(
#         csv_file,
#         index=False
#     )

#     print(f"\nParticipant-level values saved to:\n{csv_file}")

#%% Load and plot topomaps

import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# Folder where the CSVs live
# file_dir = r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state_only\Stats_final\Stats_Welch_Fooof_Single_Channels_Slope_Clinical.csv"
# file_dir = r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state_only\Stats_final\Welch_Fooof_PSD_Linear.csv"
file_dir = r"Z:\Analysis\SFARI_Analysis\Block_EEG_Analysis\Resting_state_only\Stats_final\Stats_eBOSC_Single_Channels_Slope_Clinical.csv"

df = pd.read_csv(file_dir)

#%%

# Endophenotype trend: ASD=-1, SIB=0, TD=+1
df["G_trend"] = df["Group"].map({"ASD": -1, "SIB": 0, "TD": 1}).astype(float)

# Deviation of SIB from midpoint of TD/ASD (optional; orthogonal to trend)
# ASD=-0.5, SIB=+1, TD=-0.5  (tests "SIB special" beyond linear trend)
df["G_sibdev"] = df["Group"].map({"ASD": -0.5, "SIB": 1.0, "TD": -0.5}).astype(float)

a = df.columns

#%%

df["AbsAlpha_log"] = np.log10(df["AbsAlpha"])
df["AbsTheta_log"] = np.log10(df["AbsTheta"])
df["AbsLowBeta_log"] = np.log10(df["AbsLowBeta"])
df["AbsHighBeta_log"] = np.log10(df["AbsHighBeta"])

#%% Test the slope

import statsmodels.formula.api as smf

# Choose your outcome metric here:
metric = "AvgDuration"
dfm = df.dropna(subset=[metric, "G_trend", "Subject","Age"]).copy()

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

#%% To include sex and age

# import statsmodels.formula.api as smf

# metric = "RelAlpha"
# dfm = df.dropna(subset=[metric, "G_trend", "Subject", "Age", "Sex"]).copy()
# dfm["Sex"] = dfm["Sex"].astype("category")

# dfm["Sex"] = dfm["Sex"].astype(str).str.strip()
# dfm["Sex"] = pd.Categorical(dfm["Sex"], categories=["Male", "Female"])

# md = smf.mixedlm(
#     f"{metric} ~ G_trend * Sex + Age + Region + Hemisphere + "
#     f"G_trend:Region + G_trend:Hemisphere",
#     data=dfm,
#     groups=dfm["Subject"],
#     re_formula="1"
# )

# m = md.fit(method="lbfgs")
# print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "Abundance"

dfm = df.dropna(subset=[metric, "Group", "Subject","Age"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Age + Region + Hemisphere + C(Group):Region + C(Group):Hemisphere + C(Group):Age",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())

#%%

import statsmodels.formula.api as smf

metric = "Abundance"

dfm = df.dropna(subset=[metric, "Group", "Subject"]).copy()

md = smf.mixedlm(
    f"{metric} ~ C(Group) + Region + C(Group):Region",
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(method="lbfgs")
print(m.summary())

#%% Test Right versus Left separately within every region

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

metric = "AbsAlpha_log"

required_columns = [
    metric,
    "G_trend",
    "Subject",
    "Region",
    "Hemisphere"
]

dfm = df.dropna(
    subset=required_columns
).copy()

# Make sure G_trend is numeric
dfm["G_trend"] = pd.to_numeric(
    dfm["G_trend"],
    errors="coerce"
)

# Left will be the hemisphere reference
dfm["Hemisphere"] = pd.Categorical(
    dfm["Hemisphere"],
    categories=["Left", "Right"]
)

regions = [
    "Centroparietal",
    "Frontal",
    "Occipital"
]

simple_effect_results = []

for reference_region in regions:

    df_region = dfm.copy()

    # Put the selected region first so it becomes the reference
    region_order = (
        [reference_region]
        + [
            region for region in regions
            if region != reference_region
        ]
    )

    df_region["Region"] = pd.Categorical(
        df_region["Region"],
        categories=region_order
    )

    # Same model as the original analysis
    model = smf.mixedlm(
        (
            f"{metric} ~ "
            "G_trend + "
            "Region + "
            "Hemisphere + "
            "G_trend:Region + "
            "Hemisphere:Region"
        ),
        data=df_region,
        groups=df_region["Subject"],
        re_formula="1"
    )

    result = model.fit(
        method="lbfgs",
        reml=True
    )

    # Because Left is the reference, this is Right minus Left
    term = "Hemisphere[T.Right]"

    estimate = result.params[term]
    standard_error = result.bse[term]
    z_value = result.tvalues[term]
    p_value = result.pvalues[term]

    confidence_interval = result.conf_int().loc[term]

    simple_effect_results.append({
        "Region": reference_region,
        "Contrast": "Right - Left",
        "Estimate": estimate,
        "SE": standard_error,
        "z": z_value,
        "p_uncorrected": p_value,
        "CI_2.5%": confidence_interval.iloc[0],
        "CI_97.5%": confidence_interval.iloc[1]
    })

# Convert results to dataframe
df_simple_effects = pd.DataFrame(
    simple_effect_results
)

# Holm correction for the three regional comparisons
reject, p_holm, _, _ = multipletests(
    df_simple_effects["p_uncorrected"],
    alpha=0.05,
    method="holm"
)

df_simple_effects["p_Holm"] = p_holm
df_simple_effects["Significant_Holm"] = reject

print("\nRight versus Left within each region:")
print(
    df_simple_effects.round(4).to_string(
        index=False
    )
)

#%% LMM and plot the age effect by group

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# ============================================================
# Model
# ============================================================

metric = "IAF_flat_bary"

required_columns = [
    metric,
    "Group",
    "Subject",
    "Age",
    "Region",
    "Hemisphere"
]

dfm = df.dropna(
    subset=required_columns
).copy()

dfm["Group"] = dfm["Group"].astype(str)
dfm["Subject"] = dfm["Subject"].astype(str)
dfm["Region"] = dfm["Region"].astype(str)
dfm["Hemisphere"] = dfm["Hemisphere"].astype(str)

md = smf.mixedlm(
    (
        f"{metric} ~ "
        "C(Group) + Age + Region + Hemisphere + "
        "C(Group):Region + "
        "C(Group):Hemisphere"
    ),
    data=dfm,
    groups=dfm["Subject"],
    re_formula="1"
)

m = md.fit(
    method="lbfgs",
    reml=True
)

print(m.summary())

# ============================================================
# One individual point per participant
# ============================================================

# Average all available channels/regions within each participant.
# Age and Group are constant within participant.
subject_values = (
    dfm
    .groupby(
        ["Subject", "Group"],
        as_index=False
    )
    .agg(
        Age=("Age", "first"),
        value=(metric, "mean"),
        n_observations=(metric, "count")
    )
)

print("\nParticipants by group:")
print(
    subject_values
    .groupby("Group")["Subject"]
    .nunique()
)

# ============================================================
# Model-adjusted age predictions
# ============================================================

groups = ["TD", "ASD", "SIB"]

# Only retain groups present in the data
groups = [
    group for group in groups
    if group in dfm["Group"].unique()
]

regions = sorted(
    dfm["Region"].dropna().unique()
)

hemispheres = sorted(
    dfm["Hemisphere"].dropna().unique()
)

prediction_results = []

for group in groups:

    group_subjects = subject_values[
        subject_values["Group"] == group
    ]

    # Restrict each line to the age range observed in that group
    age_grid = np.linspace(
        group_subjects["Age"].min(),
        group_subjects["Age"].max(),
        100
    )

    for age in age_grid:

        # Generate every Region × Hemisphere combination
        prediction_grid = pd.DataFrame([
            {
                "Group": group,
                "Age": age,
                "Region": region,
                "Hemisphere": hemisphere
            }
            for region in regions
            for hemisphere in hemispheres
        ])

        # Fixed-effect predictions from the fitted LMM
        prediction_grid["predicted"] = m.predict(
            prediction_grid
        )

        # Marginal prediction: equal average across regions
        # and hemispheres
        marginal_prediction = (
            prediction_grid["predicted"].mean()
        )

        prediction_results.append({
            "Group": group,
            "Age": age,
            "predicted": marginal_prediction
        })

predictions = pd.DataFrame(
    prediction_results
)

# ============================================================
# Plot
# ============================================================

group_colors = {
    "TD": "#377EB8",
    "ASD": "#E41A1C",
    "SIB": "#4DAF4A"
}

fig, ax = plt.subplots(
    figsize=(8, 6)
)

# Individual participant values
sns.scatterplot(
    data=subject_values,
    x="Age",
    y="value",
    hue="Group",
    hue_order=groups,
    palette=group_colors,
    s=65,
    alpha=0.75,
    edgecolor="white",
    linewidth=0.7,
    ax=ax
)

# Model-adjusted age lines
for group in groups:

    group_predictions = predictions[
        predictions["Group"] == group
    ]

    ax.plot(
        group_predictions["Age"],
        group_predictions["predicted"],
        color=group_colors[group],
        linewidth=2.5,
        label=f"{group} model"
    )

ax.set_xlabel(
    "Age (years)",
    fontsize=12
)

ax.set_ylabel(
    "Aperiodic-adjusted IAF (Hz)",
    fontsize=12
)

ax.set_title(
    "Association between age and individual alpha frequency",
    fontsize=14
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25
)

# Create a cleaner legend without duplicate entries
handles, labels = ax.get_legend_handles_labels()

unique_legend = {}

for handle, label in zip(handles, labels):
    # Remove the " model" suffix so points and lines
    # are represented by a single group label
    clean_label = label.replace(" model", "")
    unique_legend[clean_label] = handle

ax.legend(
    unique_legend.values(),
    unique_legend.keys(),
    title="Group",
    frameon=False
)

plt.tight_layout()
plt.show()

#%% Without AGE

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

#%% Only for Median AGE 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy

metric = "AbsAlpha_log"  # or use your variable
# dfm = the dataframe used to fit the model
# m   = fitted MixedLM results

regions = ["Centroparietal", "Frontal", "Occipital"]
hemis   = ["Left", "Right"]
groups  = ["ASD", "SIB", "TD"]

g_trend_map = {"ASD": -1.0, "SIB": 0.0, "TD": 1.0}

# Choose the Age value at which to compute EMMs
age_ref = float(dfm["Age"].mean())   # or median()

def build_exog_from_model(mixedlm_result, new_df):
    design_info = mixedlm_result.model.data.design_info
    exog = patsy.build_design_matrices([design_info], new_df, return_type="dataframe")[0]
    return exog

def predict_fe_mean_ci(mixedlm_result, x_row):
    beta = mixedlm_result.fe_params.values

    fe_names = mixedlm_result.model.exog_names
    cov = mixedlm_result.cov_params()
    cov_fe = cov.loc[fe_names, fe_names].values if hasattr(cov, "loc") else cov[:len(beta), :len(beta)]

    x = np.asarray(x_row).reshape(1, -1)
    mu = float(x @ beta)
    se = float(np.sqrt(x @ cov_fe @ x.T))
    ci = 1.96 * se
    return mu, mu - ci, mu + ci

rows = []
for g in groups:
    for r in regions:
        # Hemisphere-averaged prediction: average the design rows for Left & Right
        exogs = []
        for h in hemis:
            tmp = pd.DataFrame([{
                "G_trend": g_trend_map[g],
                "Age": age_ref,
                "Region": r,
                "Hemisphere": h,
            }])
            exog = build_exog_from_model(m, tmp)
            exogs.append(exog.values[0])

        xbar = np.mean(np.vstack(exogs), axis=0)
        mu, lo, hi = predict_fe_mean_ci(m, xbar)

        rows.append({"Group": g, "Region": r, "Pred": mu, "CI_low": lo, "CI_high": hi})

emm = pd.DataFrame(rows)

# Plot
group_order  = ["ASD", "SIB", "TD"]
region_order = ["Centroparietal", "Occipital", "Frontal"]
xpos = {g:i for i,g in enumerate(group_order)}

plt.figure(figsize=(8.2, 4.8))
for r in region_order:
    sub = emm[emm["Region"] == r].copy()
    sub["x"] = sub["Group"].map(xpos)
    plt.plot(sub["x"], sub["Pred"], marker="o", linewidth=2, label=r)
    yerr = np.vstack([sub["Pred"] - sub["CI_low"], sub["CI_high"] - sub["Pred"]])
    plt.errorbar(sub["x"], sub["Pred"], yerr=yerr, fmt="none", capsize=6, linewidth=2)

plt.xticks(range(len(group_order)), group_order)
plt.ylabel(f"Model-predicted {metric} (mean ± 95% CI)")
plt.title(f"MixedLM EMMs at Age = {age_ref:.2f} (averaged over hemispheres)")
plt.legend(frameon=False, title="Region")
plt.tight_layout()
plt.show()

#%% Age Effect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy

# Age range
age_min = dfm["Age"].min()
age_max = dfm["Age"].max()
age_grid = np.linspace(age_min, age_max, 50)

regions = ["Centroparietal", "Frontal", "Occipital"]
hemis = ["Left", "Right"]

# Choose a group (e.g., ASD reference)
g_trend_ref = -1.0   # ASD

rows = []

for r in regions:
    for age in age_grid:
        exogs = []
        for h in hemis:
            tmp = pd.DataFrame([{
                "G_trend": g_trend_ref,
                "Age": age,
                "Region": r,
                "Hemisphere": h,
            }])
            exog = patsy.build_design_matrices(
                [m.model.data.design_info],
                tmp,
                return_type="dataframe"
            )[0]
            exogs.append(exog.values[0])

        xbar = np.mean(np.vstack(exogs), axis=0)
        mu = float(xbar @ m.fe_params.values)

        rows.append({
            "Age": age,
            "Region": r,
            "Pred": mu
        })

dev_df = pd.DataFrame(rows)

plt.figure(figsize=(7,5))

for r in regions:
    sub = dev_df[dev_df["Region"] == r]
    plt.plot(sub["Age"], sub["Pred"], linewidth=2, label=r)

plt.xlabel("Age")
plt.ylabel("Model-predicted AbsAlpha_log")
plt.title("Developmental effect on resting alpha (ASD reference)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

#%%

groups = {"ASD": -1.0, "SIB": 0.0, "TD": 1.0}

rows = []

for g_name, g_val in groups.items():
    for age in age_grid:
        exogs = []
        for h in hemis:
            tmp = pd.DataFrame([{
                "G_trend": g_val,
                "Age": age,
                "Region": "Centroparietal",   # or pick a region
                "Hemisphere": h,
            }])
            exog = patsy.build_design_matrices(
                [m.model.data.design_info],
                tmp,
                return_type="dataframe"
            )[0]
            exogs.append(exog.values[0])

        xbar = np.mean(np.vstack(exogs), axis=0)
        mu = float(xbar @ m.fe_params.values)

        rows.append({
            "Age": age,
            "Group": g_name,
            "Pred": mu
        })

dev_group = pd.DataFrame(rows)

plt.figure(figsize=(7,5))

for g in dev_group["Group"].unique():
    sub = dev_group[dev_group["Group"] == g]
    plt.plot(sub["Age"], sub["Pred"], linewidth=2, label=g)

plt.xlabel("Age")
plt.ylabel("Model-predicted AbsAlpha_log")
plt.title("Developmental trajectories by group (Frontal)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

#%%

import pandas as pd

# Average within subject × region
subj_region = (
    dfm
    .groupby(["Subject", "Group", "Region"])[metric]
    .mean()
    .reset_index()
)

# Pivot to wide format
wide = subj_region.pivot_table(
    index=["Subject", "Group"],
    columns="Region",
    values=metric
).reset_index()

# Compute posterior–frontal gradient
wide["Parietal_minus_Frontal"] = (
    wide["Centroparietal"] - wide["Frontal"]
)

import seaborn as sns
import matplotlib.pyplot as plt

order = ['ASD','SIB','TD']

plt.figure(figsize=(6,5))
sns.boxplot(
    data=wide,
    x="Group",
    y="Parietal_minus_Frontal",
    order=order
)
sns.stripplot(
    data=wide,
    x="Group",
    y="Parietal_minus_Frontal",
    color="black",
    order=order,
    alpha=0.4
)

plt.ylabel("Parietal – Frontal Alpha (log)")
plt.title("Anterior–posterior alpha gradient by group")
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
metric = "IAF_bary"   # change to RelAlpha etc.

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

df_td = df_all[df_all['Group'] == 'TD']
df_asd = df_all[df_all['Group'] == 'ASD']
df_sib = df_all[df_all['Group'] == 'SIB']

# Filter for the channels of interest
td_filtered = df_td[df_td['Channel'].isin(channels_of_interest)]
asd_filtered = df_asd[df_asd['Channel'].isin(channels_of_interest)]
sib_filtered = df_sib[df_sib['Channel'].isin(channels_of_interest)]

# Average duration per subject across these channels
td_avg = td_filtered.groupby('Subject')['RelAlpha'].mean().reset_index()
td_avg['Group'] = 'TD'

asd_avg = asd_filtered.groupby('Subject')['RelAlpha'].mean().reset_index()
asd_avg['Group'] = 'ASD'

sib_avg = sib_filtered.groupby('Subject')['RelAlpha'].mean().reset_index()
sib_avg['Group'] = 'SIB'

# Combine the data
combined_df = pd.concat([td_avg, asd_avg, sib_avg], ignore_index=True)

# Plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=combined_df, x='Group', y='RelAlpha', palette='Set2')
sns.swarmplot(data=combined_df, x='Group', y='RelAlpha', color='black', alpha=0.7)
plt.title('Average Alpha Episode Duration\n(7–13 Hz) over Parieto-Occipital Channels')
plt.ylabel('Mean Duration (s)')
plt.xlabel('Group')
plt.tight_layout()
plt.show()

#%% One-way ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

# Fit linear model
model = ols('RelAlpha ~ C(Group)', data=combined_df).fit()

# ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)


#%%

from scipy.stats import ttest_ind

# Extract data per group
td_values = combined_df[combined_df['Group'] == 'TD']['RelAlpha'].values
asd_values = combined_df[combined_df['Group'] == 'ASD']['RelAlpha'].values

# Run independent t-test
t_stat, p_val = ttest_ind(td_values, asd_values, equal_var=False)  # Welch's t-test

print(f"T-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")

#%%

import statsmodels.formula.api as smf

model = smf.ols('RelAlpha ~ C(Group)', data=combined_df).fit()
print(model.summary())

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- combined_df must already have one row per Subject with columns: Group, RelAlpha ---

order = ["ASD", "SIB", "TD"]  # gradient order

# Ensure categorical order
dfp = combined_df.copy()
dfp["Group"] = pd.Categorical(dfp["Group"], categories=order, ordered=True)

# Numeric code for trend line (ASD=0, SIB=1, TD=2)
dfp["Group_code"] = dfp["Group"].cat.codes

# Group summaries for mean ± 95% CI
summary = (
    dfp.groupby("Group", observed=True)["RelAlpha"]
       .agg(["mean", "count", "std"])
       .reset_index()
)
summary["sem"] = summary["std"] / np.sqrt(summary["count"])
summary["ci95"] = 1.96 * summary["sem"]

# Fit simple linear trend across ordered groups (visual guide)
coef = np.polyfit(dfp["Group_code"], dfp["RelAlpha"], 1)
xline = np.linspace(dfp["Group_code"].min(), dfp["Group_code"].max(), 100)
yline = coef[0] * xline + coef[1]

# --- Plot ---
plt.figure(figsize=(7.5, 4.5))

# Individual subjects
sns.stripplot(
    data=dfp, x="Group", y="RelAlpha",
    order=order, jitter=0.22, size=4,
    alpha=0.6, color="black"
)

# Mean ± 95% CI overlay
xpos = np.arange(len(order))
plt.errorbar(
    xpos, summary["mean"], yerr=summary["ci95"],
    fmt="o", capsize=6, linewidth=2
)

# Trend line across groups (visual)
plt.plot(xline, yline, linestyle="--", linewidth=2)

plt.xlabel("")
plt.ylabel("Relative alpha power (7–13 Hz)")
plt.title("ASD → SIB → TD shift in relative alpha power\n(points = subjects; circles = mean ± 95% CI)")
plt.tight_layout()
plt.show()



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
td_data = df_td.pivot_table(index='Subject', columns='Channel', values='RelAlpha', fill_value=np.nan)
asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='RelAlpha', fill_value=np.nan)

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
# td_data = df_td.pivot_table(index='Subject', columns='Channel', values='MeanIBI', fill_value=np.nan)
# asd_data = df_asd.pivot_table(index='Subject', columns='Channel', values='MeanIBI', fill_value=np.nan)

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

# tfce_threshold = dict(start=0.0, step=0.3)

# Determine the cluster-forming threshold based on a critical t-value
n1, n2 = td_values.shape[0], asd_values.shape[0]
dfree = n1 + n2 - 2
t_threshold = t.ppf(1 - 0.05/2, dfree)  # two-sided cluster-forming threshold

# Classical permutation test
# Perform the test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    X, stat_fun=stat_fun, n_permutations=1000, threshold=t_threshold, adjacency=adjacency,out_type = 'indices',
    tail=0, n_jobs=1)

print("Best (min) TFCE-corrected p-value:", np.min(cluster_p_values) if len(cluster_p_values) else np.nan)

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
    if cluster_p_values[i] < 0.1:  # Significance threshold
        significant_indices = cluster  # Extract indices
        mask[significant_indices] = True  # Apply mask to those indices

plot_topomap(T_obs, info, mask=mask, cmap='RdBu_r', vlim=(-vlim, vlim), axes=ax, show=True)

# Add a colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('T-values')

plt.show()