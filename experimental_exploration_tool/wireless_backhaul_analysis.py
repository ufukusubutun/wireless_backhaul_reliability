from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brentq

if int(np.__version__.split(".")[0]) >= 2:
    raise ImportError(
        "wireless_backhaul_analysis requires NumPy < 2 because the current "
        "itur/astropy stack used here is not compatible with NumPy 2.x. "
        "Install compatible versions such as `numpy<2`, `astropy==6.0.1`, "
        "and `itur==0.4.0`, then restart the kernel."
    )

from itur.models import itu530


DEFAULT_PTX_EIRP_DBM = 80.0
DEFAULT_BANDWIDTHS_HZ = {
    "same": {6: 500e6, 11: 500e6, 18: 500e6, 23: 500e6, 39: 500e6, 80: 500e6},
    "diff": {6: 120e6, 11: 160e6, 18: 200e6, 23: 200e6, 39: 0.8e9, 80: 2e9},
}
DEFAULT_RX_ANT_GAIN_DB = {6: 28.9, 11: 34.2, 18: 38.4, 23: 40.6, 39: 45.2, 80: 51.4}
DEFAULT_FREQ_ORDER = ["6 GHz", "11 GHz", "18 GHz", "23 GHz", "39 GHz", "80 GHz"]


def _base_palette() -> dict[str, tuple[float, float, float]]:
    palette = sns.color_palette()
    return {
        "6 GHz": palette[0],
        "11 GHz": palette[1],
        "80 GHz": palette[2],
        "18 GHz": palette[3],
        "23 GHz": palette[4],
        "39 GHz": palette[8],
    }


DEFAULT_FREQ_COLORS = _base_palette()
DEFAULT_FREQ_COLORS.update(
    {
        "6 GHz & 39 GHz": DEFAULT_FREQ_COLORS["39 GHz"],
        "6 GHz & 80 GHz": DEFAULT_FREQ_COLORS["80 GHz"],
        "11 GHz & 39 GHz": DEFAULT_FREQ_COLORS["39 GHz"],
        "11 GHz & 80 GHz": DEFAULT_FREQ_COLORS["80 GHz"],
        "18 GHz & 39 GHz": sns.color_palette()[7],
        "18 GHz & 80 GHz": sns.color_palette()[6],
        "23 GHz & 39 GHz": sns.color_palette()[7],
        "23 GHz & 80 GHz": sns.color_palette()[7],
    }
)


RAIN_STATS_LOOKUP = {
    6: {"k": 0.0004878, "alpha": 1.5728},
    11: {"k": 0.01731, "alpha": 1.1617},
    18: {"k": 0.07708, "alpha": 1.0025},
    23: {"k": 0.1284, "alpha": 0.9630},
    28: {"k": 0.1964, "alpha": 0.9277},
    39: {"k": 0.4058, "alpha": 0.8486},
    80: {"k": 1.1668, "alpha": 0.7021},
}


@dataclass(frozen=True)
class City:
    name: str
    climate: str
    lat: float
    lon: float
    h_e: float
    h_r: float


CITY_PRESETS = {
    "Phoenix": City(
        name="Phoenix",
        climate="Dry",
        lat=33.4484,
        lon=-112.0740,
        h_e=360.0,
        h_r=360.0,
    ),
    "New York City": City(
        name="New York City",
        climate="Temperate",
        lat=40.7,
        lon=-74.0,
        h_e=80.0,
        h_r=50.0,
    ),
    "Miami": City(
        name="Miami",
        climate="Tropical",
        lat=25.8,
        lon=-80.2,
        h_e=25.0,
        h_r=25.0,
    ),
    "Panama City": City(
        name="Panama City",
        climate="Equatorial",
        lat=8.986227,
        lon=-79.522241,
        h_e=23.0,
        h_r=23.0,
    ),
}


def db2lin(db: float | np.ndarray) -> float | np.ndarray:
    return 10 ** (np.asarray(db) / 10)


def calc_capacity_gbps(prx_dBm: float, band_hz: float) -> float:
    noise_dBm = -174 + 10 * np.log10(band_hz)
    return band_hz * np.log2(1 + db2lin(prx_dBm - noise_dBm)) / 1e9


def calc_pathloss(d_km: float, fc_ghz: float) -> float:
    return 92.44 + 20 * np.log10(fc_ghz) + 20 * np.log10(d_km)


def calc_weather_attenuation(
    p_percent: float,
    d_km: float,
    fc_ghz: float,
    rain_rate001: float,
) -> float:
    rain_stats = RAIN_STATS_LOOKUP[fc_ghz]
    k = rain_stats["k"]
    alpha = rain_stats["alpha"]
    gamma = k * (rain_rate001**alpha)

    r_den = (
        0.477 * (d_km**0.633) * (rain_rate001 ** (0.073 * alpha)) * (fc_ghz**0.123)
        - 10.579 * (1 - np.exp(-0.024 * d_km))
    )
    r = 1 / r_den
    a001 = gamma * d_km * r

    c0 = 0.12
    if fc_ghz >= 10:
        c0 += 0.4 * ((np.log10(fc_ghz / 10)) ** 0.8)

    c1 = (0.07**c0) * (0.12 ** (1 - c0))
    c2 = 0.855 * c0 + 0.546 * (1 - c0)
    c3 = 0.139 * c0 + 0.043 * (1 - c0)

    return a001 * c1 * p_percent ** (-(c2 + c3 * np.log10(p_percent)))


def inverse_calc_weather_attenuation(
    target_A_dB: float,
    d_km: float,
    fc_ghz: float,
    rain_rate001: float,
    p_min: float = 1e-5 * 100,
    p_max: float = 1e-2 * 100,
) -> float:
    def root_function(p_percent: float) -> float:
        return calc_weather_attenuation(p_percent, d_km, fc_ghz, rain_rate001) - target_A_dB

    if target_A_dB < calc_weather_attenuation(p_max, d_km, fc_ghz, rain_rate001):
        return 1e2
    if target_A_dB > calc_weather_attenuation(p_min, d_km, fc_ghz, rain_rate001):
        return 1e-6
    return brentq(root_function, p_min, p_max)


def calc_Q(p_in_100: float, Q1: float = 2.85, beta: float = 0.13) -> float:
    if p_in_100 <= (Q1 / 12) ** (1 / beta):
        return 12
    if (Q1 / 12) ** (1 / beta) < p_in_100 <= 3:
        return Q1 * (p_in_100**-beta)
    if 3 < p_in_100 <= 30:
        return Q1 * (3**-beta)
    exponent = np.log(Q1 * (3**-beta)) / np.log(0.3)
    return Q1 * (3**-beta) * ((p_in_100 / 30) ** exponent)


def multipath_fading(
    lat: float,
    lon: float,
    h_e: float,
    h_r: float,
    d_km: float,
    fc_GHz: float,
    A_dB: float,
) -> float:
    p0 = itu530.multipath_loss_for_A(
        lat=lat,
        lon=lon,
        h_e=h_e,
        h_r=h_r,
        d=d_km,
        f=fc_GHz,
        A=0,
    ).value
    At = 25 + 1.2 * np.log10(p0)

    if A_dB >= At:
        pw = p0 * (10 ** (-A_dB / 10))
    else:
        pt = p0 * (10 ** (-At / 10))
        term = -np.log((100.0 - pt) / 100.0)
        qa_p = (-20.0 / At) * np.log10(term)

        factor = (1.0 + 0.3 * (10.0 ** (-At / 20.0))) * 10.0 ** (-0.016 * At)
        qt = (qa_p - 2.0) / factor - 4.3 * ((10.0 ** (-At / 20.0)) + At / 800.0)

        bracket1 = 1.0 + 0.3 * (10.0 ** (-At / 20.0))
        bracket2 = 10.0 ** (-0.016 * A_dB)
        bracket3 = qt + 4.3 * ((10.0 ** (-At / 20.0)) + A_dB / 800.0)
        qa = 2.0 + bracket1 * bracket2 * bracket3

        exponent_inner = 10.0 ** (-qa * A_dB / 20.0)
        pw = 100.0 * (1.0 - np.exp(-exponent_inner))

    Q = calc_Q(pw)
    return pw / Q / 100


def build_results_dataframe(
    cities: Sequence[City],
    distances_km: Sequence[float],
    carrier_freqs_ghz: Sequence[int],
    hops: Sequence[int],
    attenuation_grid_db: Sequence[float] | np.ndarray,
    bandwidth_mode: str = "diff",
    ptx_eirp_dBm: float = DEFAULT_PTX_EIRP_DBM,
    n_paths: int = 1,
    bandwidths_hz: dict[str, dict[int, float]] | None = None,
    rx_ant_gain_db: dict[int, float] | None = None,
    min_outage: float = 1e-6,
    max_outage: float = 5e-2,
) -> pd.DataFrame:
    from itur.models.itu837 import rainfall_rate

    bandwidths_hz = bandwidths_hz or DEFAULT_BANDWIDTHS_HZ
    rx_ant_gain_db = rx_ant_gain_db or DEFAULT_RX_ANT_GAIN_DB

    rows: list[dict[str, float | str | int]] = []
    attenuation_grid_db = np.asarray(attenuation_grid_db)

    for city in cities:
        rain_rate001 = rainfall_rate(city.lat, city.lon, 0.01).value

        for distance_km in distances_km:
            for fc_ghz in carrier_freqs_ghz:
                for hop_count in hops:
                    per_hop_distance_km = distance_km / hop_count

                    for A_dB in attenuation_grid_db:
                        pathloss_dB = calc_pathloss(per_hop_distance_km, fc_ghz)

                        if per_hop_distance_km >= 5:
                            multipath_outage = multipath_fading(
                                city.lat,
                                city.lon,
                                city.h_e,
                                city.h_r,
                                per_hop_distance_km,
                                fc_ghz,
                                A_dB,
                            )
                        else:
                            multipath_outage = 0.0

                        rain_outage = inverse_calc_weather_attenuation(
                            target_A_dB=A_dB,
                            d_km=per_hop_distance_km,
                            fc_ghz=fc_ghz,
                            rain_rate001=rain_rate001,
                        ) / 100

                        total_outage = 1 - (1 - (rain_outage + multipath_outage)) ** hop_count
                        total_outage = float(np.clip(total_outage, 0.0, 1.0))

                        prx_dBm = ptx_eirp_dBm + rx_ant_gain_db[fc_ghz] - pathloss_dB - A_dB
                        capacity_gbps = calc_capacity_gbps(
                            prx_dBm=prx_dBm,
                            band_hz=bandwidths_hz[bandwidth_mode][fc_ghz],
                        )

                        rows.append(
                            {
                                "city": city.name,
                                "climate": city.climate,
                                "city_title": f"{city.name} - {city.climate}",
                                "rain_label": f"{city.name} - {rain_rate001:.2f} mm/h",
                                "lat": city.lat,
                                "lon": city.lon,
                                "h_e": city.h_e,
                                "h_r": city.h_r,
                                "rain_rate001_mm_per_hr": rain_rate001,
                                "d_km": float(distance_km),
                                "per_hop_d_km": float(per_hop_distance_km),
                                "carrier_freq_ghz": fc_ghz,
                                "carrier_label": f"{fc_ghz} GHz",
                                "capacity_gbps": capacity_gbps,
                                "bandwidth_mode": bandwidth_mode,
                                "bandwidth_hz": bandwidths_hz[bandwidth_mode][fc_ghz],
                                "N_h": hop_count,
                                "N_p": n_paths,
                                "A_dB": float(A_dB),
                                "rain_outage": rain_outage,
                                "multipath_outage": multipath_outage,
                                "p": total_outage,
                            }
                        )

    df = pd.DataFrame(rows)
    df = df[(df["p"] > min_outage) & (df["p"] < max_outage)].copy()
    df["nines"] = -np.log10(df["p"])
    df["availability_pct"] = 100 * (1 - df["p"])
    return df.sort_values(
        ["city", "d_km", "carrier_freq_ghz", "N_h", "nines"]
    ).reset_index(drop=True)


def combine_bands_interp(
    df: pd.DataFrame,
    combos: Sequence[Sequence[str]],
    x_col: str = "nines",
    scenario_cols: Sequence[str] | None = None,
    freq_col: str = "carrier_label",
    cap_col: str = "capacity_gbps",
    label_sep: str = " & ",
    n_points: int = 200,
) -> pd.DataFrame:
    if scenario_cols is None:
        ignored = {x_col, freq_col, cap_col}
        scenario_cols = [column for column in df.columns if column not in ignored]

    keep_cols = list(scenario_cols) + [x_col, freq_col, cap_col]
    df2 = df[keep_cols].copy()
    out_rows: list[dict[str, object]] = []

    for key_vals, group in df2.groupby(list(scenario_cols), dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_dict = dict(zip(scenario_cols, key_vals))

        for combo in combos:
            sub = group[group[freq_col].isin(combo)]
            if sub.empty:
                continue

            bands_present = set(sub[freq_col].unique())
            if not all(band in bands_present for band in combo):
                continue

            band_data = {}
            x_mins = []
            x_maxs = []
            valid = True

            for band in combo:
                band_group = sub[sub[freq_col] == band].sort_values(x_col)
                xs = band_group[x_col].to_numpy()
                ys = band_group[cap_col].to_numpy()

                if len(xs) < 2:
                    valid = False
                    break

                band_data[band] = (xs, ys)
                x_mins.append(xs.min())
                x_maxs.append(xs.max())

            if not valid:
                continue

            x_min = max(x_mins)
            x_max = min(x_maxs)
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                continue

            x_common = np.linspace(x_min, x_max, n_points)
            cap_sum = np.zeros_like(x_common)
            for band in combo:
                xs, ys = band_data[band]
                cap_sum += np.interp(x_common, xs, ys)

            combo_label = label_sep.join(combo)
            for x_value, y_value in zip(x_common, cap_sum):
                row = dict(key_dict)
                row[x_col] = x_value
                row[freq_col] = combo_label
                row[cap_col] = y_value
                out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(out_rows)


def build_augmented_band_dataframe(
    df: pd.DataFrame,
    combos: Sequence[Sequence[str]],
    keep_single_labels: Sequence[str] | None = None,
    scenario_cols: Sequence[str] | None = None,
    x_col: str = "nines",
    freq_col: str = "carrier_label",
    cap_col: str = "capacity_gbps",
    label_sep: str = " & ",
    n_points: int = 200,
) -> pd.DataFrame:
    combined_df = combine_bands_interp(
        df=df,
        combos=combos,
        x_col=x_col,
        scenario_cols=scenario_cols,
        freq_col=freq_col,
        cap_col=cap_col,
        label_sep=label_sep,
        n_points=n_points,
    )

    frames = []
    if keep_single_labels:
        frames.append(df[df[freq_col].isin(keep_single_labels)].copy())
    if not combined_df.empty:
        frames.append(combined_df)
    if not frames:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(frames, ignore_index=True)


def select_scenario(
    df: pd.DataFrame,
    city: str | None = None,
    distance_km: float | None = None,
    hops: int | None = None,
    bandwidth_mode: str | None = None,
    carrier_labels: Sequence[str] | None = None,
    n_paths: int | None = 1,
) -> pd.DataFrame:
    subset = df.copy()
    if city is not None:
        subset = subset[subset["city"] == city]
    if distance_km is not None:
        subset = subset[np.isclose(subset["d_km"], distance_km)]
    if hops is not None:
        subset = subset[subset["N_h"] == hops]
    if bandwidth_mode is not None:
        subset = subset[subset["bandwidth_mode"] == bandwidth_mode]
    if carrier_labels is not None:
        subset = subset[subset["carrier_label"].isin(carrier_labels)]
    if n_paths is not None and "N_p" in subset.columns:
        subset = subset[subset["N_p"] == n_paths]
    return subset.copy()


def summarize_capacity_at_outages(
    df: pd.DataFrame,
    outage_probabilities: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5),
    group_cols: Sequence[str] = ("city", "d_km", "carrier_label", "N_h"),
    x_col: str = "nines",
    y_col: str = "capacity_gbps",
) -> pd.DataFrame:
    target_nines = np.array([-np.log10(outage) for outage in outage_probabilities])
    label_map = {
        outage: f"{100 * (1 - outage):.3f}%".rstrip("0").rstrip(".") for outage in outage_probabilities
    }

    rows: list[dict[str, object]] = []
    for key_vals, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(group_cols, key_vals))

        ordered = group.sort_values(x_col)
        xs = ordered[x_col].to_numpy()
        ys = ordered[y_col].to_numpy()
        if len(xs) < 2:
            continue

        for outage, nine in zip(outage_probabilities, target_nines):
            column_label = label_map[outage]
            if nine < xs.min() or nine > xs.max():
                row[column_label] = np.nan
            else:
                row[column_label] = float(np.interp(nine, xs, ys))
        rows.append(row)

    return pd.DataFrame(rows)


def style_capacity_axis(
    ax: plt.Axes,
    title: str,
    x_limits: tuple[float, float] = (2, 5),
    y_limits: tuple[float, float] = (-0.4, 14),
    reference_capacity_gbps: float | None = 3.0,
    reference_outage: float | None = 5e-5,
    xlabel: str = "Reliability level beta",
    ylabel: str = "Guaranteed Capacity zeta* (Gbps)",
) -> None:
    ax.grid(True)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks([2, 3, 4, 5], ["99%", "99.9%", "99.99%", "99.999%"])
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.tick_params(axis="both", labelsize=10)

    if reference_outage is not None:
        ax.axvline(-np.log10(reference_outage), linestyle="--", color="black", alpha=0.35, zorder=1)
    if reference_capacity_gbps is not None:
        ax.axhline(reference_capacity_gbps, linestyle="--", color="black", alpha=0.35, zorder=1)


def plot_city_panels(
    df: pd.DataFrame,
    cities: Sequence[str],
    distance_km: float,
    bandwidth_mode: str = "diff",
    hops: int = 1,
    carrier_labels: Sequence[str] | None = None,
    palette: dict[str, tuple[float, float, float]] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    palette = palette or DEFAULT_FREQ_COLORS
    figsize = figsize or (3.75 * len(cities), 3.6)
    fig, axes = plt.subplots(1, len(cities), sharey=True, figsize=figsize)
    axes = np.atleast_1d(axes)

    for idx, (ax, city) in enumerate(zip(axes, cities)):
        subset = select_scenario(
            df,
            city=city,
            distance_km=distance_km,
            hops=hops,
            bandwidth_mode=bandwidth_mode,
            carrier_labels=carrier_labels,
        )

        sns.lineplot(
            data=subset,
            x="nines",
            y="capacity_gbps",
            hue="carrier_label",
            hue_order=carrier_labels,
            lw=3,
            ax=ax,
            palette=palette,
        )

        if subset.empty:
            title = city
        else:
            title = subset["city_title"].iloc[0]

        style_capacity_axis(ax, title=title)
        if idx < len(cities) - 1 and ax.get_legend() is not None:
            ax.get_legend().remove()

    if axes[-1].get_legend() is not None:
        axes[-1].legend(title=None, fontsize=9, title_fontsize=9)

    plt.tight_layout()
    return fig, axes


def plot_single_and_combined_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    city: str,
    distance_km: float,
    title: str,
    single_labels: Sequence[str],
    combined_labels: Sequence[str],
    bandwidth_mode: str = "diff",
    hops: int = 1,
    palette: dict[str, tuple[float, float, float]] | None = None,
    x_limits: tuple[float, float] = (3, 5),
    y_limits: tuple[float, float] = (-0.4, 14),
) -> None:
    palette = palette or DEFAULT_FREQ_COLORS
    subset = select_scenario(
        df,
        city=city,
        distance_km=distance_km,
        hops=hops,
        bandwidth_mode=bandwidth_mode,
    )

    single_df = subset[subset["carrier_label"].isin(single_labels)]
    combined_df = subset[subset["carrier_label"].isin(combined_labels)]

    if not single_df.empty:
        sns.lineplot(
            data=single_df,
            x="nines",
            y="capacity_gbps",
            hue="carrier_label",
            hue_order=list(single_labels),
            lw=3,
            ax=ax,
            palette=palette,
        )

    if not combined_df.empty:
        sns.lineplot(
            data=combined_df,
            x="nines",
            y="capacity_gbps",
            hue="carrier_label",
            hue_order=list(combined_labels),
            lw=3,
            linestyle=(4, (6.4, 1.6, 1.0, 1.6)),
            ax=ax,
            palette=palette,
        )

    style_capacity_axis(
        ax,
        title=title,
        x_limits=x_limits,
        y_limits=y_limits,
        reference_capacity_gbps=None,
        reference_outage=None,
    )
    if ax.get_legend() is not None:
        ax.legend(
            title=None,
            loc="upper right",
            columnspacing=0.94,
            handlelength=1.6,
            fontsize=9,
            title_fontsize=9,
        )


def plot_multihop_vs_combining_panel(
    ax: plt.Axes,
    base_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    city: str,
    distance_km: float,
    target_band: str,
    combo_label: str,
    title: str,
    hop_values: Sequence[int] = (1, 2, 3),
    bandwidth_mode: str = "diff",
    palette: dict[str, tuple[float, float, float]] | None = None,
    x_limits: tuple[float, float] = (2.7, 5),
    y_limits: tuple[float, float] = (-0.4, 14),
) -> None:
    palette = palette or DEFAULT_FREQ_COLORS
    hop_linestyles = {
        1: "-",
        2: (0, (3, 1)),
        3: ":",
        4: (0, (5, 1)),
    }

    base_subset = select_scenario(
        base_df,
        city=city,
        distance_km=distance_km,
        bandwidth_mode=bandwidth_mode,
        carrier_labels=[target_band],
        n_paths=1,
    )
    base_subset = base_subset[base_subset["N_h"].isin(hop_values)]

    if not base_subset.empty:
        band_color = palette[target_band]
        for hop_value in hop_values:
            hop_subset = base_subset[base_subset["N_h"] == hop_value]
            if hop_subset.empty:
                continue
            sns.lineplot(
                data=hop_subset,
                x="nines",
                y="capacity_gbps",
                lw=3,
                color=band_color,
                linestyle=hop_linestyles.get(hop_value, "-"),
                ax=ax,
            )

    combo_subset = select_scenario(
        combined_df,
        city=city,
        distance_km=distance_km,
        hops=1,
        bandwidth_mode=bandwidth_mode,
        carrier_labels=[combo_label],
        n_paths=1,
    )

    if not combo_subset.empty:
        sns.lineplot(
            data=combo_subset,
            x="nines",
            y="capacity_gbps",
            lw=3,
            linestyle="-.",
            ax=ax,
            color=palette[combo_label],
        )

    style_capacity_axis(
        ax,
        title=title,
        x_limits=x_limits,
        y_limits=y_limits,
        reference_capacity_gbps=None,
        reference_outage=None,
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()
