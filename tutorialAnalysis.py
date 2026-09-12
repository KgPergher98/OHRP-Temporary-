"""
tutorialAnalysis.py
===================

Turning the raw CSVs under `results/` into the figures and tables reported in
the dissertation.

    python tutorialAnalysis.py

This script does not itself produce the figures. It explains the two production
scripts that do, checks whether the inputs they need are present, and prints the
exact sequence to run. Keeping the heavy lifting in the original scripts means
what you run here is the same code that generated the published material.

--------------------------------------------------------------------------
The two production scripts
--------------------------------------------------------------------------
analise_experimentos.py
    Reads  results/<experiment>_Ts_WL_*.csv   (daily return series)
           results/<experiment>_Ms_WL_*.csv   (per-portfolio measures)
    Writes paper/figures/series_temporais/    cumulative return, one per WL,
                                              gross (solid) vs net of cost (dashed)
           paper/figures/distribuicoes/       per-metric distributions across
                                              the 89 out-of-sample portfolios
           paper/figures/barras_anuais/       annual bar charts per metric
           paper/figures/correlacoes/         cross-strategy correlation heatmaps
           paper/tables/                      summary, Wilcoxon and t-test tables

    Pick the experiment at the top of the file, in the EXPERIMENTS block:
        1 -> ExperimentoSOHRP  (short windows, six strategies)
        2 -> ExperimentoOHRP   (long windows, four strategies)
    Run it once per experiment; re-running overwrites the shared output folders.

setores_b3.py
    Reads  datasets/BR_equity_closing_prices.csv
           datasets/SectoralBrazilianClassification.xlsx
    Writes paper/figures/setores/pie_sector.{pdf,png}
    Sector composition of the investable universe. Needs no simulation results,
    so it can be run at any time.

--------------------------------------------------------------------------
Full sequence, from an empty checkout
--------------------------------------------------------------------------
    python simulationFund.py        # hours: fills results/
    python analise_experimentos.py  # with EXPERIMENT_TO_ANALYSE = 1
    python analise_experimentos.py  # then with EXPERIMENT_TO_ANALYSE = 2
    python setores_b3.py
"""

#%%
import glob
import os

RESULTS_DIR = "results"
EXPERIMENTS = ("ExperimentoSOHRP", "ExperimentoOHRP")

EXPECTED = {
    "ExperimentoSOHRP": [round(0.2 * i, 2) for i in range(1, 11)],
    "ExperimentoOHRP": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
}


def inventory():
    """Report which simulation outputs are already available."""
    print(f"Looking for simulation output in ./{RESULTS_DIR}/\n")
    ready = True

    for experiment in EXPERIMENTS:
        measures = sorted(glob.glob(
            os.path.join(RESULTS_DIR, f"{experiment}_Ms_WL_*.csv")))
        series = sorted(glob.glob(
            os.path.join(RESULTS_DIR, f"{experiment}_Ts_WL_*.csv")))
        expected = len(EXPECTED[experiment])

        status = "complete" if len(measures) == expected else "incomplete"
        print(f"  {experiment}")
        print(f"      measures files : {len(measures):>2} / {expected}   ({status})")
        print(f"      series files   : {len(series):>2} / {expected}")

        if len(measures) < expected:
            ready = False
            missing = [wl for wl in EXPECTED[experiment]
                       if not os.path.exists(
                           os.path.join(RESULTS_DIR,
                                        f"{experiment}_Ms_WL_{wl:.2f}.csv"))]
            shown = ", ".join(f"{wl:.2f}" for wl in missing[:6])
            more = "" if len(missing) <= 6 else f" (+{len(missing) - 6} more)"
            print(f"      missing WL     : {shown}{more}")
        print()

    return ready


if __name__ == "__main__":
    ready = inventory()

    if ready:
        print("All simulation output is present. You can now run:\n")
        print("    python analise_experimentos.py   # once per experiment")
        print("    python setores_b3.py\n")
    else:
        print("Simulation output is missing, so the analysis scripts have "
              "nothing to read.\n")
        print("Run the backtest first:\n")
        print("    python simulationFund.py\n")
        print("It writes one pair of CSVs per window length and skips any that "
              "already exist,\nso an interrupted run can simply be relaunched.")
        print("\nFor a quick look at the mechanics without the full cost, see "
              "tutorialSimulation.py.")

    print("\nNote: setores_b3.py depends only on the datasets and can be run "
          "right away.")

# %%
