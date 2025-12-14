import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CSV_FILE = "521426916_Deliverable01_DataSet.csv" 

df = pd.read_csv(CSV_FILE)

print("Columns in your CSV:")
print(df.columns)


# COLUMN NAME MAPPINGS
STOP_COL    = "stop_name"
BOARD_COL   = "boarded"
ALIGHT_COL  = "alighted"
OCC_COL     = "occupancy_after"
DWELL_COL   = "dwell_time_s"
HEADWAY_COL = "headway_s"
TRIP_COL    = "trip_id"

# Helper function to safely check columns
def check_cols(required_cols, context_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"In {context_name}, these columns are missing from the CSV: {missing}. "
                       f"Please edit the mapping section at the top of the script.")


# TERMINAL STATISTICS 

try:
    check_cols([BOARD_COL, DWELL_COL, HEADWAY_COL, OCC_COL, STOP_COL], "Statistics Block")

    df_stats = df.copy()

    # Filter invalid values safely
    df_stats = df_stats[df_stats[HEADWAY_COL] > 0]
    df_stats = df_stats[df_stats[DWELL_COL] >= 0]

    print("\n" + "="*55)
    print("DESCRIPTIVE STATISTICS")
    print("="*55)

    print(f"Records: {len(df_stats)}")
    if TRIP_COL in df_stats.columns:
        print(f"Unique trips (trip_id): {df_stats[TRIP_COL].nunique()}")
    print(f"Unique stops: {df_stats[STOP_COL].nunique()}")

    print("\n--- Passenger Activity ---")
    print(f"Boarded:   min={df_stats[BOARD_COL].min():.0f}, max={df_stats[BOARD_COL].max():.0f}, mean={df_stats[BOARD_COL].mean():.2f}")
    if ALIGHT_COL in df_stats.columns:
        print(f"Alighted:  min={df_stats[ALIGHT_COL].min():.0f}, max={df_stats[ALIGHT_COL].max():.0f}, mean={df_stats[ALIGHT_COL].mean():.2f}")

    print("\n--- Operational Times ---")
    print(f"Dwell time (s): min={df_stats[DWELL_COL].min():.1f}, max={df_stats[DWELL_COL].max():.1f}, mean={df_stats[DWELL_COL].mean():.2f}")
    print(f"Headway (s):    min={df_stats[HEADWAY_COL].min():.1f}, max={df_stats[HEADWAY_COL].max():.1f}, mean={df_stats[HEADWAY_COL].mean():.2f}")

    print("\n--- Occupancy ---")
    print(f"Occupancy after stop: min={df_stats[OCC_COL].min():.0f}, max={df_stats[OCC_COL].max():.0f}, mean={df_stats[OCC_COL].mean():.2f}")

    # M/M/1 PARAMETER SUMMARY 

    print("\n" + "="*55)
    print("M/M/1 SUMMARY")
    print("="*55)

    total_boarded = df_stats[BOARD_COL].sum()
    total_headway_min = df_stats[HEADWAY_COL].sum() / 60.0

    if total_headway_min > 0:
        lambda_per_min_overall = total_boarded / total_headway_min
    else:
        lambda_per_min_overall = np.nan

    mean_dwell_min = df_stats[DWELL_COL].mean() / 60.0
    mu_per_min_overall = (1.0 / mean_dwell_min) if mean_dwell_min > 0 else np.nan

    rho_overall = (
        lambda_per_min_overall / mu_per_min_overall
        if (np.isfinite(mu_per_min_overall) and mu_per_min_overall > 0)
        else np.nan
    )

    # ASCII-safe prints (no Greek letters)
    print(f"Estimated arrival rate (lambda) overall: {lambda_per_min_overall:.3f} passengers/min")
    print(f"Estimated service rate (mu) overall:     {mu_per_min_overall:.3f} services/min (from mean dwell time)")
    print(f"Utilization (rho) overall:               {rho_overall:.3f}")

    if np.isfinite(rho_overall):
        if rho_overall < 1:
            print("Status: STABLE (rho < 1) under overall-average approximation")
        else:
            print("Status: UNSTABLE / SATURATED (rho >= 1) under overall-average approximation")

    
    # TOP BOTTLENECK STOPS BY rho
    
    print("\n" + "="*55)
    print("TOP 5 STOPS BY UTILIZATION (rho)  [Bottleneck Candidates]")
    print("="*55)

    per_stop = df_stats.groupby(STOP_COL).agg(
        total_boarded=(BOARD_COL, "sum"),
        total_headway_s=(HEADWAY_COL, "sum"),
        avg_dwell_s=(DWELL_COL, "mean"),
        avg_occ=(OCC_COL, "mean")
    ).reset_index()

    per_stop["lambda_per_min"] = per_stop["total_boarded"] / (per_stop["total_headway_s"] / 60.0)
    per_stop["mu_per_min"] = 1.0 / (per_stop["avg_dwell_s"] / 60.0)
    per_stop["rho"] = per_stop["lambda_per_min"] / per_stop["mu_per_min"]

    top = per_stop.sort_values("rho", ascending=False).head(5)

    for _, r in top.iterrows():
        print(
            f"{r[STOP_COL]} | rho={r['rho']:.2f} | lambda={r['lambda_per_min']:.2f}/min | mu={r['mu_per_min']:.2f}/min "
            f"| avg_dwell={r['avg_dwell_s']:.1f}s | avg_occ={r['avg_occ']:.1f}"
        )

except Exception as e:
    print("\nStatistics block error:", e)


print("\n" + "="*55)


# GRAPH 1 — Arrival Rate λ per Stop

try:
    check_cols([STOP_COL, BOARD_COL, HEADWAY_COL], "Graph 1")

    df_valid = df[df[HEADWAY_COL] > 0].copy()
    arrival_stats = df_valid.groupby(STOP_COL).agg(
        total_boarded=(BOARD_COL, "sum"),
        total_headway_s=(HEADWAY_COL, "sum")
    ).reset_index()

    arrival_stats["lambda_per_min"] = arrival_stats["total_boarded"] / (arrival_stats["total_headway_s"] / 60)

    plt.figure(figsize=(10, 5))
    plt.bar(arrival_stats[STOP_COL], arrival_stats["lambda_per_min"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Arrival Rate λ (passengers/minute)")
    plt.xlabel("Stop")
    plt.title("Passenger Arrival Rate per Stop")
    plt.tight_layout()
    plt.savefig("graph1_arrival_rate_per_stop.png", dpi=300)
    plt.show()
    print("Saved: graph1_arrival_rate_per_stop.png")
except Exception as e:
    print("Graph 1 error:", e)


# GRAPH 2 — Dwell Time vs Passengers Boarding
try:
    check_cols([BOARD_COL, DWELL_COL], "Graph 2")

    plt.figure(figsize=(7, 5))
    plt.scatter(df[BOARD_COL], df[DWELL_COL])
    plt.xlabel("Passengers Boarded")
    plt.ylabel("Dwell Time (seconds)")
    plt.title("Dwell Time vs Passengers Boarding")
    plt.tight_layout()
    plt.savefig("graph2_dwell_vs_boarded.png", dpi=300)
    plt.show()
    print("Saved: graph2_dwell_vs_boarded.png")
except Exception as e:
    print("Graph 2 error:", e)


# GRAPH 3 — Headway Variability Over Sequence
try:
    check_cols([HEADWAY_COL], "Graph 3")

    if TRIP_COL in df.columns:
        df_sorted = df.sort_values([TRIP_COL])
    else:
        df_sorted = df.copy()

    plt.figure(figsize=(10, 4))
    plt.plot(df_sorted[HEADWAY_COL].values)
    plt.xlabel("Observation Index")
    plt.ylabel("Headway (seconds)")
    plt.title("Headway Variability Over Observations")
    plt.tight_layout()
    plt.savefig("graph3_headway_variability.png", dpi=300)
    plt.show()
    print("Saved: graph3_headway_variability.png")
except Exception as e:
    print("Graph 3 error:", e)


# GRAPH 4 — Occupancy Trend by Stop
try:
    check_cols([STOP_COL, OCC_COL], "Graph 4")

    occ_stats = df.groupby(STOP_COL)[OCC_COL].mean().reset_index()
    x = np.arange(len(occ_stats[STOP_COL]))

    plt.figure(figsize=(10, 4))
    plt.plot(x, occ_stats[OCC_COL], marker="o")
    plt.xticks(x, occ_stats[STOP_COL], rotation=45, ha="right")
    plt.xlabel("Stop")
    plt.ylabel("Average Occupancy After Stop")
    plt.title("Occupancy Trend by Stop")
    plt.tight_layout()
    plt.savefig("graph4_occupancy_trend.png", dpi=300)
    plt.show()
    print("Saved: graph4_occupancy_trend.png")
except Exception as e:
    print("Graph 4 error:", e)


# GRAPH 5 — Utilization Heat Map (ρ = λ / μ)
try:
    check_cols([STOP_COL, DWELL_COL], "Graph 5 (service stats)")
    if "arrival_stats" not in locals():
        check_cols([STOP_COL, BOARD_COL, HEADWAY_COL], "Graph 5 (recompute arrival_stats)")
        df_valid = df[df[HEADWAY_COL] > 0].copy()
        arrival_stats = df_valid.groupby(STOP_COL).agg(
            total_boarded=(BOARD_COL, "sum"),
            total_headway_s=(HEADWAY_COL, "sum")
        ).reset_index()
        arrival_stats["lambda_per_min"] = arrival_stats["total_boarded"] / (arrival_stats["total_headway_s"] / 60)

    service_stats = df.groupby(STOP_COL).agg(
        avg_dwell_s=(DWELL_COL, "mean")
    ).reset_index()

    merged = arrival_stats.merge(service_stats, on=STOP_COL)
    merged["mu_per_min"] = 1.0 / (merged["avg_dwell_s"] / 60.0)
    merged["rho"] = merged["lambda_per_min"] / merged["mu_per_min"]

    rho_row = merged["rho"].values.reshape(1, -1)

    plt.figure(figsize=(10, 2))
    plt.imshow(rho_row, aspect="auto")
    plt.xticks(range(len(merged[STOP_COL])), merged[STOP_COL], rotation=45, ha="right")
    plt.yticks([])
    plt.colorbar(label="Utilization ρ")
    plt.title("Utilization Heat Map by Stop")
    plt.tight_layout()
    plt.savefig("graph5_utilization_heatmap.png", dpi=300)
    plt.show()
    print("Saved: graph5_utilization_heatmap.png")
except Exception as e:
    print("Graph 5 error:", e)



# GRAPH 6 — Queueing Model Diagram (Conceptual)

try:
    plt.figure(figsize=(10, 3))
    plt.text(0.1, 0.5, "Passenger Arrival (λ)", fontsize=12)
    plt.text(0.35, 0.5, "QUEUE", fontsize=14, bbox=dict(boxstyle="round", fc="white"))
    plt.text(0.6, 0.5, "BUS SERVICE\n(μ)", fontsize=14, bbox=dict(boxstyle="round", fc="white"))
    plt.text(0.85, 0.5, "Departures", fontsize=12)

    plt.annotate("", xy=(0.32, 0.5), xytext=(0.2, 0.5), arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(0.57, 0.5), xytext=(0.45, 0.5), arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(0.83, 0.5), xytext=(0.72, 0.5), arrowprops=dict(arrowstyle="->"))

    plt.axis("off")
    plt.title("Queueing System Representation of a Bus Stop")
    plt.tight_layout()
    plt.savefig("graph6_queueing_diagram.png", dpi=300)
    plt.show()
    print("Saved: graph6_queueing_diagram.png")
except Exception as e:
    print("Graph 6 error:", e)


# GRAPH 7 — Bottleneck Visualization
try:
    check_cols([STOP_COL, BOARD_COL, DWELL_COL, OCC_COL], "Graph 7")

    bottleneck_stats = df.groupby(STOP_COL).agg(
        total_boarded=(BOARD_COL, "sum"),
        avg_dwell_s=(DWELL_COL, "mean"),
        avg_occupancy=(OCC_COL, "mean")
    ).reset_index()

    for col in ["total_boarded", "avg_dwell_s", "avg_occupancy"]:
        min_v = bottleneck_stats[col].min()
        max_v = bottleneck_stats[col].max()
        if max_v > min_v:
            bottleneck_stats[col + "_norm"] = (bottleneck_stats[col] - min_v) / (max_v - min_v)
        else:
            bottleneck_stats[col + "_norm"] = 0.0

    x = np.arange(len(bottleneck_stats[STOP_COL]))

    plt.figure(figsize=(10, 5))
    plt.plot(x, bottleneck_stats["total_boarded_norm"], marker="o", label="Boarding (norm)")
    plt.plot(x, bottleneck_stats["avg_dwell_s_norm"], marker="s", label="Dwell time (norm)")
    plt.plot(x, bottleneck_stats["avg_occupancy_norm"], marker="^", label="Occupancy (norm)")
    plt.xticks(x, bottleneck_stats[STOP_COL], rotation=45, ha="right")
    plt.ylabel("Normalized value (0–1)")
    plt.xlabel("Stop")
    plt.title("Bottleneck Indicators by Stop")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graph7_bottlenecks.png", dpi=300)
    plt.show()
    print("Saved: graph7_bottlenecks.png")
except Exception as e:
    print("Graph 7 error:", e)


# GRAPH 8 — Queue Length vs Utilization (M/M/1)
try:
    rho_vals = np.linspace(0.1, 0.95, 50)
    Lq = (rho_vals ** 2) / (1 - rho_vals)

    plt.figure(figsize=(7, 5))
    plt.plot(rho_vals, Lq)
    plt.xlabel("Utilization ρ")
    plt.ylabel("Expected Queue Length Lq")
    plt.title("Queue Length vs Utilization (M/M/1)")
    plt.tight_layout()
    plt.savefig("graph8_Lq_vs_rho.png", dpi=300)
    plt.show()
    print("Saved: graph8_Lq_vs_rho.png")
except Exception as e:
    print("Graph 8 error:", e)


# GRAPH 9 — Waiting Time vs Utilization (M/M/1)
try:
    rho_vals = np.linspace(0.1, 0.95, 50)
    Wq = rho_vals / (1 - rho_vals)

    plt.figure(figsize=(7, 5))
    plt.plot(rho_vals, Wq)
    plt.xlabel("Utilization ρ")
    plt.ylabel("Waiting Time Wq (normalized units)")
    plt.title("Waiting Time vs Utilization (M/M/1)")
    plt.tight_layout()
    plt.savefig("graph9_Wq_vs_rho.png", dpi=300)
    plt.show()
    print("Saved: graph9_Wq_vs_rho.png")
except Exception as e:
    print("Graph 9 error:", e)


print("\nAll graphs processed. Check for any 'error' messages above and fix column names if needed.")
