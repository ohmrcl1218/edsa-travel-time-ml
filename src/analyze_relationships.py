
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

# Assumes this file is located inside the src folder.
ROOT = Path(__file__).resolve().parent.parent

DATA_PATHS = [
    ROOT / "data" / "processed" / "edsa_training_data.csv",
    ROOT / "data" / "processed" / "edsa_dataset.csv",
    ROOT / "data" / "edsa_dataset.csv",
]

RESULTS_DIR = ROOT / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

data_path = next((path for path in DATA_PATHS if path.exists()), None)

if data_path is None:
    raise FileNotFoundError(
        "Dataset not found. Check that your CSV is located at:\n"
        f"{DATA_PATHS[0]}"
    )

df = pd.read_csv(data_path)

# Remove accidental spaces from column names.
df.columns = df.columns.str.strip()

print("=" * 60)
print("EDSA TRAVEL TIME RELATIONSHIP ANALYSIS")
print("=" * 60)
print(f"Dataset: {data_path}")
print(f"Number of records: {len(df)}")
print(f"Number of columns: {len(df.columns)}")


# ============================================================
# 3. STANDARDIZE COMMON COLUMN NAMES
# ============================================================

# Maps common dataset column names to consistent names.
column_aliases = {
    "travel_time_min": "travel_time_minutes",
    "travel_time": "travel_time_minutes",
    "avg_speed_kph": "average_speed_kmh",
    "avg_speed_kmh": "average_speed_kmh",
    "average_speed_kph": "average_speed_kmh",
    "traffic_volume_veh": "traffic_volume",
    "traffic_density_veh_per_km": "traffic_density",
    "hour_of_day": "hour",
}

rename_columns = {}

for original, standard in column_aliases.items():
    if original in df.columns and standard not in df.columns:
        rename_columns[original] = standard

df = df.rename(columns=rename_columns)

# Check that the target variable exists.
target = "travel_time_minutes"

if target not in df.columns:
    raise ValueError(
        f"Could not find the target column '{target}'.\n"
        f"Available columns: {list(df.columns)}"
    )

# Convert relevant numerical columns to numeric values.
numeric_candidates = [
    target,
    "average_speed_kmh",
    "traffic_volume",
    "traffic_density",
    "distance_km",
    "hour",
]

for column in numeric_candidates:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Parse timestamps if available.
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    # Use the timestamp to fill missing time-related information.
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    else:
        df["hour"] = df["hour"].fillna(
            df["timestamp"].dt.hour
        )

    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.day_name()
    else:
        missing_day = df["day_of_week"].isna()
        df.loc[missing_day, "day_of_week"] = (
            df.loc[missing_day, "timestamp"].dt.day_name()
        )

# Convert numerical day-of-week values to names when appropriate.
if "day_of_week" in df.columns:
    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    def normalize_day(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.integer, np.floating)):
            if float(value).is_integer():
                number = int(value)

                # Supports both 0-6 and 1-7 weekday encodings.
                if 0 <= number <= 6:
                    return day_names[number]
                if number == 7:
                    return "Sunday"

        text = str(value).strip().title()

        for day in day_names.values():
            if text == day or text.startswith(day[:3]):
                return day

        return text

    df["day_name"] = df["day_of_week"].apply(normalize_day)

elif "timestamp" in df.columns:
    df["day_name"] = df["timestamp"].dt.day_name()

# Derive a time period from the hour if one is not available.
if "period" not in df.columns and "hour" in df.columns:

    def assign_period(hour):
        if pd.isna(hour):
            return np.nan
        if 6 <= hour <= 9:
            return "AM"
        if 10 <= hour <= 15:
            return "OFF"
        if 16 <= hour <= 19:
            return "PM"
        return "OTHER"

    df["period"] = df["hour"].apply(assign_period)

# Remove records without a usable target value.
df = df.dropna(subset=[target]).copy()

if df.empty:
    raise ValueError("No valid travel-time records remain after cleaning.")

print(f"Records with valid travel time: {len(df)}")
print(f"Columns available: {list(df.columns)}")


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

descriptive_columns = [
    column
    for column in [
        target,
        "average_speed_kmh",
        "traffic_volume",
        "traffic_density",
        "distance_km",
    ]
    if column in df.columns
]

descriptive_stats = df[descriptive_columns].describe().T
descriptive_stats.to_csv(
    RESULTS_DIR / "descriptive_statistics.csv"
)

print(descriptive_stats.round(3))


# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SPEARMAN CORRELATION WITH TRAVEL TIME")
print("=" * 60)

correlation_columns = [
    column
    for column in [
        target,
        "average_speed_kmh",
        "traffic_volume",
        "traffic_density",
        "distance_km",
        "hour",
    ]
    if column in df.columns
]

correlation_data = df[correlation_columns].corr(method="spearman")

# Save the full Spearman correlation matrix.
correlation_data.to_csv(RESULTS_DIR / "correlation_matrix.csv")

# Save each variable's Spearman correlation with travel time.
correlation_results = (
    correlation_data[target]
    .drop(index=target)
    .rename("spearman_correlation")
    .reset_index()
    .rename(columns={"index": "variable"})
)

correlation_results["absolute_correlation"] = (
    correlation_results["spearman_correlation"].abs()
)

correlation_results = correlation_results.sort_values(
    "absolute_correlation", ascending=False
)

correlation_results["method"] = "spearman"

correlation_results.to_csv(
    RESULTS_DIR / "correlation_results.csv",
    index=False
)

print(correlation_results.round(4).to_string(index=False))

if "traffic_volume" in df.columns:
    spearman_traffic = (
        df[[target, "traffic_volume"]]
        .corr(method="spearman")
        .loc[target, "traffic_volume"]
    )

    spearman_result = pd.DataFrame(
        [
            {
                "variable": "traffic_volume",
                "spearman_correlation": spearman_traffic,
                "absolute_correlation": abs(spearman_traffic),
                "method": "spearman",
            }
        ]
    )

    spearman_result.to_csv(
        RESULTS_DIR / "traffic_volume_vs_travel_time_spearman.csv",
        index=False
    )

    print("\nSpearman correlation for travel time vs traffic volume:")
    print(spearman_result.round(4).to_string(index=False))


# ============================================================
# 6. ONE-WAY ANOVA: DAY OF THE WEEK AND TIME OF DAY
# ============================================================

if "day_name" in df.columns:
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    day_groups = [
        df.loc[df["day_name"] == day, target].dropna()
        for day in day_order
        if day in df["day_name"].unique()
    ]

    if len(day_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*day_groups)

        day_anova = pd.DataFrame(
            [
                {
                    "factor": "day_of_week",
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "group_count": len(day_groups),
                }
            ]
        )

        day_anova.to_csv(
            RESULTS_DIR / "anova_day_of_week.csv",
            index=False,
        )

        print("\n" + "=" * 60)
        print("ONE-WAY ANOVA: TRAVEL TIME BY DAY OF THE WEEK")
        print("=" * 60)
        print(day_anova.round(6).to_string(index=False))
    else:
        print("\nNo valid ANOVA comparison could be run for day of the week.")

if "period" in df.columns:
    period_order = ["AM", "OFF", "PM", "OTHER"]

    period_groups = [
        df.loc[df["period"] == period, target].dropna()
        for period in period_order
        if period in df["period"].unique()
    ]

    if len(period_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*period_groups)

        period_anova = pd.DataFrame(
            [
                {
                    "factor": "time_of_day",
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "group_count": len(period_groups),
                }
            ]
        )

        period_anova.to_csv(
            RESULTS_DIR / "anova_time_of_day.csv",
            index=False,
        )

        print("\n" + "=" * 60)
        print("ONE-WAY ANOVA: TRAVEL TIME BY TIME OF DAY")
        print("=" * 60)
        print(period_anova.round(6).to_string(index=False))
    else:
        print("\nNo valid ANOVA comparison could be run for time of day.")


# ============================================================
# 7. MEAN TRAVEL TIME BY TIME PERIOD
# ============================================================

if "period" in df.columns:
    period_stats = (
        df.dropna(subset=["period"])
        .groupby("period", observed=True)[target]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    period_stats = period_stats.rename(
        columns={
            "count": "record_count",
            "mean": "mean_travel_time_min",
            "median": "median_travel_time_min",
            "std": "std_travel_time_min",
        }
    )

    period_stats.to_csv(
        RESULTS_DIR / "mean_travel_time_by_period.csv",
        index=False
    )

    print("\nMean travel time by period:")
    print(period_stats.round(3).to_string(index=False))

    # Bar chart: mean travel time by period.
    plot_data = period_stats.copy()

    period_order = ["AM", "OFF", "PM", "OTHER"]
    plot_data["_order"] = plot_data["period"].astype(str).apply(
        lambda value: (
            period_order.index(value)
            if value in period_order
            else len(period_order)
        )
    )
    plot_data = plot_data.sort_values("_order")

    plt.figure(figsize=(8, 5))
    plt.bar(
        plot_data["period"].astype(str),
        plot_data["mean_travel_time_min"]
    )
    plt.xlabel("Time Period")
    plt.ylabel("Mean Travel Time (minutes)")
    plt.title("Mean EDSA Travel Time by Time Period")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "travel_time_by_period.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 7. MEAN TRAVEL TIME BY DAY OF THE WEEK
# ============================================================

if "day_name" in df.columns:
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    day_stats = (
        df.dropna(subset=["day_name"])
        .groupby("day_name", observed=True)[target]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    day_stats = day_stats.rename(
        columns={
            "count": "record_count",
            "mean": "mean_travel_time_min",
            "median": "median_travel_time_min",
            "std": "std_travel_time_min",
        }
    )

    day_stats["_order"] = day_stats["day_name"].apply(
        lambda value: (
            day_order.index(value)
            if value in day_order
            else len(day_order)
        )
    )

    day_stats = day_stats.sort_values("_order")
    day_stats.drop(columns="_order").to_csv(
        RESULTS_DIR / "mean_travel_time_by_day.csv",
        index=False
    )

    print("\nMean travel time by day of the week:")
    print(
        day_stats.drop(columns="_order")
        .round(3)
        .to_string(index=False)
    )

    # Bar chart: mean travel time by day.
    plt.figure(figsize=(10, 5))
    plt.bar(
        day_stats["day_name"],
        day_stats["mean_travel_time_min"]
    )
    plt.xlabel("Day of the Week")
    plt.ylabel("Mean Travel Time (minutes)")
    plt.title("Mean EDSA Travel Time by Day of the Week")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "travel_time_by_day.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 8. SCATTER PLOT: SPEED VS. TRAVEL TIME
# ============================================================

if "average_speed_kmh" in df.columns:
    plot_data = df[
        ["average_speed_kmh", target]
    ].dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        plot_data["average_speed_kmh"],
        plot_data[target],
        alpha=0.6
    )

    # Add a linear trend line when at least two distinct
    # speed values are available.
    if (
        len(plot_data) >= 2
        and plot_data["average_speed_kmh"].nunique() >= 2
    ):
        x = plot_data["average_speed_kmh"].to_numpy()
        y = plot_data[target].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        plt.plot(x_line, y_line)

    plt.xlabel("Average Vehicle Speed (km/h)")
    plt.ylabel("Travel Time (minutes)")
    plt.title("Average Vehicle Speed vs. EDSA Travel Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "speed_vs_travel_time.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 9. SCATTER PLOT: TRAFFIC VOLUME VS. TRAVEL TIME
# ============================================================

if "traffic_volume" in df.columns:
    plot_data = df[
        ["traffic_volume", target]
    ].dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        plot_data["traffic_volume"],
        plot_data[target],
        alpha=0.6
    )

    if (
        len(plot_data) >= 2
        and plot_data["traffic_volume"].nunique() >= 2
    ):
        x = plot_data["traffic_volume"].to_numpy()
        y = plot_data[target].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        plt.plot(x_line, y_line)

    plt.xlabel("Traffic Volume (vehicles)")
    plt.ylabel("Travel Time (minutes)")
    plt.title("Traffic Volume vs. EDSA Travel Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "traffic_volume_vs_travel_time.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 10. OPTIONAL: MEAN TRAVEL TIME BY HOUR
# ============================================================

if "hour" in df.columns:
    hourly_stats = (
        df.dropna(subset=["hour"])
        .groupby("hour")[target]
        .agg(["count", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "record_count",
                "mean": "mean_travel_time_min",
            }
        )
        .sort_values("hour")
    )

    hourly_stats.to_csv(
        RESULTS_DIR / "mean_travel_time_by_hour.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        hourly_stats["hour"],
        hourly_stats["mean_travel_time_min"],
        marker="o"
    )
    plt.xlabel("Hour of the Day (0–23)")
    plt.ylabel("Mean Travel Time (minutes)")
    plt.title("Mean EDSA Travel Time by Hour")
    plt.xticks(range(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "travel_time_by_hour.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 11. FINISH
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)
print(f"CSV results saved in: {RESULTS_DIR}")
print(f"Graphs saved in: {GRAPH_DIR}")

print("\nGenerated graphs:")
for graph in sorted(GRAPH_DIR.glob("*.png")):
    print(f"- {graph.name}")

print("\nImportant: Spearman measures monotonic rank association, which is appropriate for this travel-time analysis.")
print("Correlation alone does not prove causation.")
