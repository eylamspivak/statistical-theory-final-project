"""
NBA Career Longevity - Statistical Analysis
Eilam Spivak | Asaf Buzaglo
Statistical Theory, Bar-Ilan University, 2026

METHODOLOGY OVERVIEW:
  - Incident-cohort filtering: the dataset begins at the 1996-97 season,
    so a player whose first *recorded* row is that season cannot be
    confirmed as a true rookie (they may already have been active
    earlier, outside the data's coverage). We restrict the analytic
    sample to players whose first recorded season is strictly later,
    for whom the recorded debut season is confirmed genuine.
  - career_length is defined as the span (in seasons) from a player's
    first to last appearance in the dataset; we also compute the
    alternative distinct-season-count definition and report how many
    players the two definitions disagree on (players with a gap
    season), to make this choice explicit rather than implicit.
  - Player rows are grouped by earliest season using position-based
    selection (sort + take first row) so that all columns for a given
    player consistently come from the same season's record.
  - Two-sample comparisons use Welch's t-test (unequal variances),
    matching the standard-error formula used for confidence intervals
    and appropriate given the variance heterogeneity confirmed by
    Levene's test in the draft-round comparison.
  - p-values are computed via the survival function for numerical
    stability at very small values.
  - Goodness-of-fit uses a discrete chi-square test (grouping exact
    integer values, appropriate for the small-integer, heavily-tied
    career_length variable) as the primary test, alongside a
    Geometric-distribution MLE fit and an empirical hazard-rate
    calculation that visualizes departures from the constant-hazard
    assumption directly.
  - Levene's test (variance homogeneity) and Kruskal-Wallis (the
    non-parametric counterpart to ANOVA) are reported alongside the
    one-way ANOVA for the draft-round comparison.
  - Post-hoc statistical power and a power curve are reported for the
    PPG group comparison.
  - A nested regression model comparison (rookie-season attributes /
    draft-group / both combined) with a Generalized Likelihood Ratio
    Test framing directly tests whether performance and draft
    circumstance each carry independent explanatory power.
  - The Wald Sequential Probability Ratio Test is run across 1000
    random re-orderings of the data, reporting the full distribution
    of stopping times and decisions rather than a single sequence,
    since a single ordering is not representative on its own.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, chi2_contingency, chi2,
    kstest, shapiro, f_oneway, pearsonr, spearmanr,
    levene, kruskal, nct
)
from itertools import combinations
import warnings
# NOTE: warnings are not blanket-suppressed here, so that legitimate
# ties/precision warnings from KS and other tests remain visible.

# Requires: pip install scikit-learn --break-system-packages
#           (for classification & cross-validation)

# ── Style ────────────────────────────────────────────────────────────
NAVY   = "#0D1B2A"
GOLD   = "#C9A84C"
ACCENT = "#00B4D8"
LIGHT  = "#E8EDF2"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   LIGHT,
    "axes.edgecolor":   NAVY,
    "axes.labelcolor":  NAVY,
    "xtick.color":      NAVY,
    "ytick.color":      NAVY,
    "text.color":       NAVY,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.labelsize":   12,
})


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def fmt_p(p):
    """Consistent p-value formatting; avoids printing a misleading 0.0000."""
    if p < 0.0001:
        return "p<0.0001"
    return f"p={p:.4f}"


def two_sided_t_p(t_stat, df):
    """Numerically stable two-sided p-value using the survival function."""
    return 2 * stats.t.sf(np.abs(t_stat), df=df)


# ════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ════════════════════════════════════════════════════════════════════

def load_data(path="all_seasons.csv", first_dataset_year=1996):
    """
    Load NBA dataset from Kaggle (justinas/nba-players-data).
    Download 'all_seasons.csv' and place it next to this script.

    IMPORTANT FIX (incident-cohort filtering):
    The dataset only starts at the 1996-97 season. A player whose first
    *recorded* row is 1996-97 could actually be a veteran who debuted
    years earlier (left-truncation) -- we simply cannot tell from the
    data. Treating that row as their "rookie season" would silently
    corrupt draft_age and every downstream analysis that uses it.
    We therefore drop any player whose first appearance in the dataset
    IS the dataset's first season, keeping only the "incident cohort"
    of players we can confidently identify as true rookies.
    """
    df = pd.read_csv(path)
    df["year"] = df["season"].str[:4].astype(int)

    # ---- Two career-length definitions; compare AFTER filtering ------
    # (divergence is reported on the actual analytic sample below, not
    # on the raw pre-filter data, so the stat matches the sample size
    # readers see everywhere else in the paper)
    span = (df.groupby("player_name")["year"]
              .agg(lambda x: x.max() - x.min() + 1))
    nuniq = df.groupby("player_name")["year"].nunique()
    career = span.reset_index().rename(columns={"year": "career_length"})

    # ---- FIX: groupby().first() can mix rows across columns when ---
    # ---- there are per-column NaNs; use position-based head(1) -----
    first = (df.sort_values("year")
               .groupby("player_name", as_index=False)
               .head(1)
               .reset_index(drop=True))

    data = first.merge(career, on="player_name")

    # ---- FIX: incident-cohort filter (left-truncation) --------------
    n_before = len(data)
    data = data[data["year"] > first_dataset_year].copy()
    n_after = len(data)
    print(f"  ℹ️  Incident-cohort filter: dropped {n_before - n_after} "
          f"players whose first recorded season was {first_dataset_year}"
          f"-{str(first_dataset_year+1)[2:]} (dataset's first season) -- "
          f"cannot confirm these were true rookie seasons. "
          f"{n_after} players remain.")

    # Now report the span-vs-nunique divergence on the ANALYTIC (filtered)
    # sample specifically, not the raw pre-filter data.
    analytic_players = set(data["player_name"])
    span_f = span[span.index.isin(analytic_players)]
    nuniq_f = nuniq[nuniq.index.isin(analytic_players)]
    diff_players = (span_f != nuniq_f).sum()
    print(f"  ℹ️  career-length definitions (within the {n_after}-player "
          f"analytic sample): span vs. distinct-season-count differ for "
          f"{diff_players} players (players with a gap season). "
          f"Using SPAN (first-to-last, inclusive) as career_length "
          f"throughout, consistent with the paper.")

    # Rename for clarity
    rename = {
        "pts":           "PPG",
        "reb":           "RPG",
        "ast":           "APG",
        "net_rating":    "net_rtg",
        "usg_pct":       "usg",
        "player_height": "height_cm",
        "player_weight": "weight_kg",
        "age":           "draft_age",
    }
    data = data.rename(columns={k: v for k, v in rename.items() if k in data.columns})

    # Clean draft_round: Kaggle data often stores "Undrafted" as text
    if "draft_round" in data.columns:
        data["draft_round_clean"] = pd.to_numeric(data["draft_round"], errors="coerce")
        data["draft_group"] = np.where(
            data["draft_round_clean"] == 1, "Round 1",
            np.where(data["draft_round_clean"].isna(), "Undrafted", "Round 2+")
        )

    data = data.dropna(subset=["career_length"])
    data["career_length"] = data["career_length"].astype(int)

    # NOTE (limitation): player_name is used as the merge key because
    # the dataset has no player_id; two different players who happen to
    # share a name would be incorrectly merged into one. We did not find
    # evidence of this in the dataset, but flag it as a known limitation.

    print(f"✅  Final sample: {len(data)} players | "
          f"career range: {data.career_length.min()}–{data.career_length.max()} seasons")
    if "draft_age" in data.columns:
        print(f"  ℹ️  draft_age summary after filtering: "
              f"mean={data['draft_age'].mean():.2f}, "
              f"SD={data['draft_age'].std():.2f} "
              f"(sanity check: true NBA rookie-age SD is typically ~1.5-1.7; "
              f"a much larger value here would indicate residual "
              f"left-truncation contamination)")
    return data


# ════════════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE STATS
# ════════════════════════════════════════════════════════════════════

def descriptive_stats(data):
    print("\n" + "═"*55)
    print("  DESCRIPTIVE STATISTICS – career_length")
    print("═"*55)
    cl = data["career_length"]
    print(f"  n        = {len(cl)}")
    print(f"  Mean     = {cl.mean():.2f}")
    print(f"  Median   = {cl.median():.2f}")
    print(f"  Std      = {cl.std():.2f}")
    print(f"  Min/Max  = {cl.min()} / {cl.max()}")
    print(f"  Skewness = {cl.skew():.3f}")
    print(f"  Kurtosis = {cl.kurt():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Career Length – Distribution", fontsize=15, color=NAVY, fontweight="bold")

    axes[0].hist(cl, bins=20, color=NAVY, edgecolor=GOLD, linewidth=0.8)
    axes[0].axvline(cl.mean(), color=GOLD, linestyle="--", linewidth=2,
                    label=f"Mean={cl.mean():.1f}")
    axes[0].set_xlabel("Career Length (seasons)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].set_title("Histogram")

    axes[1].boxplot(cl, patch_artist=True,
                    boxprops=dict(facecolor=LIGHT, color=NAVY),
                    medianprops=dict(color=GOLD, linewidth=2),
                    whiskerprops=dict(color=NAVY),
                    capprops=dict(color=NAVY))
    axes[1].set_ylabel("Career Length (seasons)")
    axes[1].set_title("Boxplot")

    plt.tight_layout()
    plt.savefig("fig1_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig1_distribution.png saved")


# ════════════════════════════════════════════════════════════════════
# 3. GOODNESS-OF-FIT  (מבחן טיב התאמה)
# ════════════════════════════════════════════════════════════════════

def goodness_of_fit(data):
    print("\n" + "═"*55)
    print("  GOODNESS-OF-FIT")
    print("═"*55)
    cl = data["career_length"].values
    n = len(cl)

    # ---- KS / Shapiro (continuous approximations) -------------------
    # CAVEAT: parameters below are estimated FROM this sample, so the
    # standard KS null distribution is not strictly valid here (this is
    # the classic Lilliefors problem) -- we report it for continuity
    # with the exam-style KS test, but treat the DISCRETE chi-square
    # test below as the primary, statistically appropriate GOF test
    # for integer-valued career_length.
    mu, sigma = np.mean(cl), np.std(cl)
    d_n, p_n = kstest(cl, "norm", args=(mu, sigma))
    d_e, p_e = kstest(cl, "expon", args=(1, cl.mean() - 1))  # loc=1 (min of data)
    sw, sw_p = shapiro(cl[:5000] if len(cl) > 5000 else cl)

    print(f"  KS – Normal:      D={d_n:.4f}, {fmt_p(p_n)}  "
          f"{'✅' if p_n>0.05 else '❌'}  "
          f"[caveat: params estimated from sample -> Lilliefors problem,")
    print(f"                    p-value is only indicative, not exact]")
    print(f"  KS – Exponential: D={d_e:.4f}, {fmt_p(p_e)}  "
          f"{'✅' if p_e>0.05 else '❌'}  (loc fixed at 1, matching data min)")
    print(f"  Shapiro-Wilk:     W={sw:.4f},  {fmt_p(sw_p)}")

    # ---- Discrete chi-square GOF: shared exact-value grouping --------
    # FIX (previous version had a binning bug): np.histogram()'s
    # half-open bins [a,b) were being compared against expected mass
    # intended for the CLOSED-integer group (a,b], creating a
    # systematic one-step offset -- verified by simulation to falsely
    # reject even genuinely geometric data (chi2 > 500 on true
    # Geometric(p) data). Fixed by counting EXACT integer values per
    # group (not histogram bins) and using matching probability mass
    # for both the Normal and Geometric expected counts, with the tail
    # groups extending to +/-infinity so probability mass sums to
    # exactly 1 (the previous version silently dropped ~12% of the
    # Normal's mass below zero).
    groups = [[1], [2], [3], [4], [5, 6], [7, 8], [9, 10, 11]]
    group_labels = ["1", "2", "3", "4", "5-6", "7-8", "9-11", "12+"]
    obs = np.array([np.isin(cl, g).sum() for g in groups] + [(cl >= 12).sum()])

    # -- Normal expected counts (edges at half-integers, +/-inf tails) --
    edges_normal = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 11.5, np.inf])
    exp_p_normal = np.diff(stats.norm.cdf(edges_normal, mu, sigma))
    exp_normal = exp_p_normal * n
    chi2_stat = np.sum((obs - exp_normal)**2 / exp_normal)
    dof = len(obs) - 1 - 2  # -2 for estimated mu, sigma
    p_chi2 = stats.chi2.sf(chi2_stat, dof)
    print(f"\n  Discrete χ² GOF vs Normal (exact-integer grouping, "
          f"infinite tails):")
    print(f"    χ²={chi2_stat:.3f}, df={dof}, {fmt_p(p_chi2)}  "
          f"{'✅ fail to reject' if p_chi2>0.05 else '❌ reject normality'}")

    # ---- Geometric distribution MLE fit ------------------------------
    # A natural model for "career length": each season, a player
    # survives to the next with probability (1-p) and exits with
    # probability p, giving P(L=k) = (1-p)^(k-1) * p, k=1,2,3,...
    # MLE: p_hat = 1 / mean(L)  (closed form for the geometric).
    p_hat = 1.0 / cl.mean()
    print(f"\n  Geometric distribution MLE fit:")
    print(f"    p_hat (season-to-season exit probability) = {p_hat:.4f}")
    print(f"    Implied median survival ≈ {np.log(0.5)/np.log(1-p_hat):.1f} seasons")

    gp = lambda k, p: (1 - p)**(k - 1) * p  # P(L=k) for geometric(p)
    probs_geom = [sum(gp(k, p_hat) for k in g) for g in groups]
    probs_geom.append(1 - sum(probs_geom))  # 12+ tail, exact (no renormalization needed)
    exp_geom = np.array(probs_geom) * n
    chi2_geom = np.sum((obs - exp_geom)**2 / exp_geom)
    dof_geom = len(obs) - 1 - 1  # -1 for estimated p
    p_geom = stats.chi2.sf(chi2_geom, dof_geom)
    print(f"    χ² GOF: χ²={chi2_geom:.3f}, df={dof_geom}, {fmt_p(p_geom)}  "
          f"{'✅ geometric fits' if p_geom>0.05 else '❌ geometric rejected — see empirical hazard curve below'}")

    # Plot: Geometric fit overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    k_vals = np.arange(1, cl.max()+1)
    geom_pmf = (1 - p_hat)**(k_vals - 1) * p_hat
    ax.hist(cl, bins=np.arange(0.5, cl.max()+1.5, 1), density=True,
            color=NAVY, edgecolor=GOLD, alpha=0.75, label="Observed")
    ax.plot(k_vals, geom_pmf, color="#B22222", linewidth=2.5, marker='o',
            markersize=3, label=f"Geometric MLE fit ($\\hat{{p}}$={p_hat:.3f})")
    ax.set_xlabel("Career Length (seasons)")
    ax.set_ylabel("Density")
    ax.set_title(f"Geometric Distribution Fit to Career Length\n"
                 f"χ²={chi2_geom:.1f}, {fmt_p(p_geom)}",
                 color=NAVY, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig15_geometric_fit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig15_geometric_fit.png saved")

    # ---- Empirical hazard curve --------------------------------------
    # h(k) = P(exit at season k | survived to season k) = #{L=k} / #{L>=k}
    # Directly shows WHY the geometric (constant-hazard) model does or
    # doesn't fit, rather than inferring it indirectly from a rejected
    # chi-square test. This is the empirical analogue of what a
    # Kaplan-Meier hazard estimate would show, without requiring
    # censoring-aware machinery (all careers here are already complete
    # in the sense of "ended within the observed window").
    k_max_hazard = int(np.percentile(cl, 95))  # avoid noisy tail with tiny risk sets
    hazard = []
    risk_set_sizes = []
    for k in range(1, k_max_hazard + 1):
        at_risk = (cl >= k).sum()
        exits = (cl == k).sum()
        risk_set_sizes.append(at_risk)
        hazard.append(exits / at_risk if at_risk > 0 else np.nan)

    print(f"\n  Empirical hazard rate h(k) = P(exit at season k | reached season k):")
    for k, h, r in zip(range(1, k_max_hazard + 1), hazard, risk_set_sizes):
        print(f"    k={k:>2}: h={h:.3f}  (risk set n={r})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ks = np.arange(1, k_max_hazard + 1)
    ax.plot(ks, hazard, color=NAVY, linewidth=2, marker='o', markersize=5,
            label="Empirical hazard $\\hat{h}(k)$")
    ax.axhline(p_hat, color="#B22222", linewidth=2, linestyle="--",
               label=f"Constant hazard implied by Geometric fit ($\\hat p$={p_hat:.3f})")
    ax.set_xlabel("Season k")
    ax.set_ylabel("Hazard rate (exit probability)")
    ax.set_title("Empirical Season-to-Season Exit Hazard\n"
                 "vs. the Constant Hazard the Geometric Model Assumes",
                 color=NAVY, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig18_hazard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig18_hazard.png saved")

    # Manual Q-Q plot (no statsmodels needed)
    fig, ax = plt.subplots(figsize=(6, 5))
    (osm, osr), (slope, intercept, r) = stats.probplot(cl, dist="norm")
    ax.scatter(osm, osr, color=NAVY, alpha=0.5, s=15, label="Data")
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, slope * line_x + intercept, color=GOLD,
            linewidth=2, label="Normal reference")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot – Career Length vs Normal",
                 color=NAVY, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig2_qqplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig2_qqplot.png saved")

    return {"p_hat_geometric": p_hat, "chi2_geom_p": p_geom}


# ════════════════════════════════════════════════════════════════════
# 4. H1 – PPG effect  (Welch t-test + Mann-Whitney + Power)
# ════════════════════════════════════════════════════════════════════

def hypothesis_ppg(data):
    print("\n" + "═"*55)
    print("  H1: High PPG → Longer Career")
    print("═"*55)
    med = data["PPG"].median()
    high = data[data["PPG"] >= med]["career_length"]
    low  = data[data["PPG"] <  med]["career_length"]
    n1, n2 = len(high), len(low)

    print(f"  Median PPG split: {med:.1f}")
    print(f"  High PPG mean career: {high.mean():.2f} (n={n1})")
    print(f"  Low  PPG mean career: {low.mean():.2f}  (n={n2})")

    # Welch's t-test (unequal variances): matches the SE formula used
    # below for the confidence interval and is appropriate given group
    # variance heterogeneity (see Levene's test in the H2 comparison).
    t, tp = ttest_ind(high, low, alternative="greater", equal_var=False)
    u, up = mannwhitneyu(high, low, alternative="greater")
    d  = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)
    diff = high.mean() - low.mean()
    se   = np.sqrt(high.var()/n1 + low.var()/n2)
    ci   = (diff - 1.96*se, diff + 1.96*se)

    print(f"\n  Welch t-test (one-sided): t={t:.3f}, {fmt_p(tp)}")
    print(f"  Mann-Whitney:             U={u:.0f},  {fmt_p(up)}")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  95% CI for diff: ({ci[0]:.2f}, {ci[1]:.2f})")

    # ---- Post-hoc power + power curve --------------------------------
    # Power of a two-sample t-test at the OBSERVED effect size, using
    # the noncentral-t distribution (exact, matches what a course-level
    # "power of the test" section should compute).
    alpha = 0.05
    n_harmonic = 2 / (1/n1 + 1/n2)  # effective n for unequal groups
    ncp = d * np.sqrt(n_harmonic / 2)
    df_t = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha, df_t)
    power_observed = 1 - nct.cdf(t_crit, df_t, ncp)
    power_note = "essentially certain to detect" if power_observed > 0.95 else (
                 "adequate (>0.8)" if power_observed > 0.8 else "underpowered (<0.8)")
    print(f"\n  Post-hoc power (at observed d={d:.2f}, α=0.05): "
          f"{power_observed:.4f}  ({power_note} an effect of this size)")

    # Power curve: what n would be needed to detect smaller effects?
    d_range = np.array([0.1, 0.2, 0.3, 0.5, round(d, 2)])
    n_range = np.arange(10, 400, 5)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_curve = [ACCENT, GOLD, "#7B3300", NAVY, "#B22222"]
    for dd, c in zip(d_range, colors_curve):
        powers = []
        for nn in n_range:
            ncp_ = dd * np.sqrt(nn / 2)
            df_ = 2*nn - 2
            tcrit_ = stats.t.ppf(1 - alpha, df_)
            powers.append(1 - nct.cdf(tcrit_, df_, ncp_))
        ax.plot(n_range, powers, color=c, linewidth=2, label=f"d={dd}")
    ax.axhline(0.8, color="black", linestyle=":", linewidth=1.5, label="Power=0.8")
    ax.set_xlabel("Sample size per group (n)")
    ax.set_ylabel("Statistical Power")
    ax.set_title("Power Curves: Required n vs. Effect Size (α=0.05, one-sided)",
                 color=NAVY, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("fig16_power_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig16_power_curve.png saved")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(low,  bins=15, alpha=0.7, color=ACCENT,
            label=f"Low PPG (<{med:.1f})", edgecolor="white")
    ax.hist(high, bins=15, alpha=0.7, color=NAVY,
            label=f"High PPG (≥{med:.1f})", edgecolor="white")
    ax.axvline(low.mean(),  color=ACCENT, linestyle="--", linewidth=2)
    ax.axvline(high.mean(), color=GOLD,   linestyle="--", linewidth=2)
    ax.set_xlabel("Career Length (seasons)")
    ax.set_ylabel("Count")
    ax.set_title(f"H1: Career Length by PPG Group (Welch t-test)\n"
                 f"t={t:.2f}, {fmt_p(tp)}, Cohen's d={d:.2f}",
                 color=NAVY, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig3_h1_ppg.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig3_h1_ppg.png saved")

    return {"power": power_observed, "d": d}


# ════════════════════════════════════════════════════════════════════
# 5. H2 – Draft Round effect (ANOVA + Levene + Kruskal-Wallis + Bonferroni)
# ════════════════════════════════════════════════════════════════════

def hypothesis_draft_round(data):
    if "draft_group" not in data.columns:
        print("\n  ⚠️  'draft_group' column not found – skipping H2")
        return

    print("\n" + "═"*55)
    print("  H2: Career Length differs by Draft Round")
    print("═"*55)

    order = [g for g in ["Round 1", "Round 2+", "Undrafted"]
             if g in data["draft_group"].values]
    groups = {g: data[data["draft_group"] == g]["career_length"].dropna()
              for g in order}

    for g, vals in groups.items():
        print(f"  {g:<10}: mean={vals.mean():.2f}, std={vals.std():.2f}, n={len(vals)}")

    # ---- FIX: check ANOVA's equal-variance assumption with Levene ---
    lev_stat, lev_p = levene(*groups.values())
    print(f"\n  Levene's test (variance homogeneity): "
          f"stat={lev_stat:.3f}, {fmt_p(lev_p)}  "
          f"{'⚠️  variances differ significantly — ANOVA assumption violated, see Kruskal-Wallis below' if lev_p<0.05 else '✅ equal-variance assumption holds'}")

    f, fp = f_oneway(*groups.values())
    print(f"\n  One-Way ANOVA: F={f:.3f}, {fmt_p(fp)}")
    print(f"  {'❌ Reject H0 – groups differ' if fp<0.05 else '✅ Fail to reject – no significant difference'}")

    # Kruskal-Wallis: the non-parametric counterpart to ANOVA, more
    # trustworthy here given the Levene result above and the
    # right-skew established in the goodness-of-fit section.
    h_stat, h_p = kruskal(*groups.values())
    print(f"  Kruskal-Wallis (non-parametric): H={h_stat:.3f}, {fmt_p(h_p)}  "
          f"{'❌ Reject H0' if h_p<0.05 else '✅ Fail to reject'}")

    # Pairwise Welch t-tests with Bonferroni correction
    pairs = list(combinations(order, 2))
    alpha_bonf = 0.05 / len(pairs)
    print(f"\n  Bonferroni α = {alpha_bonf:.4f}  ({len(pairs)} pairs, Welch t-test)")
    print(f"  {'Pair':<22} {'t':>7} {'p':>12} {'Sig?'}")
    for g1, g2 in pairs:
        t_pw, p_pw = ttest_ind(groups[g1], groups[g2], equal_var=False)
        sig = "✅" if p_pw < alpha_bonf else "—"
        print(f"  {g1} vs {g2:<10}  {t_pw:>7.3f} {fmt_p(p_pw):>12}   {sig}")

    # Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = [NAVY, GOLD, ACCENT]
    bp = ax.boxplot([groups[g] for g in order],
                    labels=order, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color(NAVY)
    ax.set_xlabel("Draft Group")
    ax.set_ylabel("Career Length (seasons)")
    ax.set_title(f"H2: Career Length by Draft Round\n"
                 f"ANOVA F={f:.2f} ({fmt_p(fp)}); Kruskal-Wallis H={h_stat:.2f} ({fmt_p(h_p)})",
                 color=NAVY, fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig("fig4_h2_draft_round.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig4_h2_draft_round.png saved")


# ════════════════════════════════════════════════════════════════════
# 6. H3 – Draft age  (Pearson + Spearman + CI)
# ════════════════════════════════════════════════════════════════════

def hypothesis_age(data):
    print("\n" + "═"*55)
    print("  H3: Draft Age negatively predicts Career Length")
    print("═"*55)
    sub = data[["draft_age", "career_length"]].dropna()
    r,  rp  = pearsonr(sub["draft_age"],  sub["career_length"])
    rho, sp = spearmanr(sub["draft_age"], sub["career_length"])

    z    = np.arctanh(r)
    se_z = 1 / np.sqrt(len(sub) - 3)
    ci   = (np.tanh(z - 1.96*se_z), np.tanh(z + 1.96*se_z))

    print(f"  Pearson  r   = {r:.4f}, {fmt_p(rp)}")
    print(f"  95% CI for r : ({ci[0]:.4f}, {ci[1]:.4f})")
    print(f"  Spearman ρ   = {rho:.4f}, {fmt_p(sp)}")
    print(f"  draft_age SD = {sub['draft_age'].std():.3f}  "
          f"(sanity check vs. true NBA rookie-age SD of ~1.5-1.7)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sub["draft_age"], sub["career_length"],
               alpha=0.35, color=NAVY, s=15)
    m, b = np.polyfit(sub["draft_age"], sub["career_length"], 1)
    xs = np.linspace(sub["draft_age"].min(), sub["draft_age"].max(), 100)
    ax.plot(xs, m*xs+b, color=GOLD, linewidth=2.5,
            label=f"OLS (slope={m:.2f})")
    ax.set_xlabel("Age at First NBA Season")
    ax.set_ylabel("Career Length (seasons)")
    ax.set_title(f"H3: Draft Age vs Career Length\n"
                 f"r={r:.3f}, {fmt_p(rp)}",
                 color=NAVY, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig5_h3_age.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig5_h3_age.png saved")


# ════════════════════════════════════════════════════════════════════
# 7. MULTIPLE LINEAR REGRESSION  (manual OLS, no statsmodels)
# ════════════════════════════════════════════════════════════════════

def _ols_fit(X_raw, y):
    """Shared OLS helper: standardizes X, fits, returns full stats dict."""
    X_mean, X_std = X_raw.mean(0), X_raw.std(0)
    X_std_safe = np.where(X_std == 0, 1, X_std)
    X_s = (X_raw - X_mean) / X_std_safe
    X   = np.column_stack([np.ones(len(X_s)), X_s])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    n, k  = len(y), X.shape[1] - 1
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2     = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    mse    = ss_res / (n - k - 1)
    var_b  = np.linalg.inv(X.T @ X) * mse
    se_b   = np.sqrt(np.diag(var_b))
    t_stat = beta / se_b
    p_vals = np.array([two_sided_t_p(t, n - k - 1) for t in t_stat])
    return {"beta": beta, "se": se_b, "t": t_stat, "p": p_vals,
            "r2": r2, "r2_adj": r2_adj, "ss_res": ss_res, "n": n, "k": k,
            "y_hat": y_hat}


def regression(data):
    print("\n" + "═"*55)
    print("  MULTIPLE LINEAR REGRESSION  (OLS, rookie-season attributes)")
    print("═"*55)

    features = [c for c in ["PPG", "RPG", "APG", "net_rtg", "usg",
                             "height_cm", "weight_kg", "draft_age"]
                if c in data.columns]
    sub = data[features + ["career_length"]].dropna()
    fit = _ols_fit(sub[features].values, sub["career_length"].values)

    print(f"\n  R² = {fit['r2']:.4f}   Adj-R² = {fit['r2_adj']:.4f}   n={fit['n']}")
    print(f"\n  {'Feature':<14} {'β_std':>8} {'SE':>8} {'t':>8} {'p':>12}")
    print("  " + "-"*54)
    names = ["const"] + features
    for i, name in enumerate(names):
        sig = " *" if fit["p"][i] < 0.05 else ""
        print(f"  {name:<14} {fit['beta'][i]:>8.4f} {fit['se'][i]:>8.4f} "
              f"{fit['t'][i]:>8.3f} {fmt_p(fit['p'][i]):>12}{sig}")

    y = sub["career_length"].values
    y_hat = fit["y_hat"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_hat, y, alpha=0.4, color=NAVY, s=15)
    lims = [min(y_hat.min(), y.min()), max(y_hat.max(), y.max())]
    ax.plot(lims, lims, color=GOLD, linewidth=2, linestyle="--", label="Perfect fit")
    ax.set_xlabel("Predicted Career Length")
    ax.set_ylabel("Actual Career Length")
    ax.set_title(f"Regression: Actual vs Predicted\nR²={fit['r2']:.3f}, Adj-R²={fit['r2_adj']:.3f}",
                 color=NAVY, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig6_regression.png", dpi=150, bbox_inches="tight")
    plt.close()

    b_feat, se_feat = fit["beta"][1:], fit["se"][1:]
    colors = [NAVY if v > 0 else "#B22222" for v in b_feat]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, b_feat, xerr=1.96*se_feat, color=colors,
            capsize=5, edgecolor="white", height=0.6)
    ax.axvline(0, color=GOLD, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Standardised Coefficient (effect on career length)")
    ax.set_title("Regression Coefficients ± 95% CI\n(rookie-season attributes: performance, physio, and draft age)",
                 color=NAVY, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig("fig7_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig6_regression.png, fig7_coefficients.png saved")
    return fit


# ════════════════════════════════════════════════════════════════════
# 7B. NESTED MODEL COMPARISON  (Performance vs Draft Round vs Both)
#     -- addresses the gap where "draft round is the strongest factor"
#     was claimed but never tested in the same model as performance.
#     The nested F-test below is algebraically a Generalized Likelihood
#     Ratio Test (GLRT) for nested linear models under normal errors.
# ════════════════════════════════════════════════════════════════════

def nested_model_comparison(data):
    print("\n" + "═"*55)
    print("  NESTED MODEL COMPARISON: Rookie Attributes vs. Draft Round")
    print("  (M1: rookie-season attributes | M2: draft only | M3: both)")
    print("═"*55)

    if "draft_group" not in data.columns:
        print("  ⚠️  'draft_group' not found – skipping")
        return

    perf_feats = [c for c in ["PPG", "RPG", "APG", "net_rtg", "usg",
                               "height_cm", "weight_kg", "draft_age"]
                  if c in data.columns]
    sub = data[perf_feats + ["draft_group", "career_length"]].dropna().copy()
    sub["is_round2"]    = (sub["draft_group"] == "Round 2+").astype(int)
    sub["is_undrafted"] = (sub["draft_group"] == "Undrafted").astype(int)
    y = sub["career_length"].values
    n = len(sub)

    X_perf  = sub[perf_feats].values
    X_draft = sub[["is_round2", "is_undrafted"]].values
    X_both  = np.column_stack([X_perf, X_draft])

    m1 = _ols_fit(X_perf,  y)   # performance only
    m2 = _ols_fit(X_draft, y)   # draft group only
    m3 = _ols_fit(X_both,  y)   # both

    print(f"\n  M1 (rookie-season attributes: performance + physio + age, k={len(perf_feats)}):  R²={m1['r2']:.4f}")
    print(f"  M2 (draft group only, k=2):             R²={m2['r2']:.4f}")
    print(f"  M3 (both, k={len(perf_feats)+2}):                  R²={m3['r2']:.4f}")

    dR2_draft = m3['r2'] - m1['r2']
    dR2_perf  = m3['r2'] - m2['r2']
    print(f"\n  ΔR² from adding draft group to performance model: {dR2_draft:+.4f}")
    print(f"  ΔR² from adding performance to draft-only model:  {dR2_perf:+.4f}")

    # ---- Nested F-test (== GLRT for nested linear models) -----------
    # F = [(SSR_reduced - SSR_full)/q] / [SSR_full/(n-k_full-1)]
    q1 = m3['k'] - m1['k']   # extra params: draft group added to M1
    F1 = ((m1['ss_res'] - m3['ss_res']) / q1) / (m3['ss_res'] / (n - m3['k'] - 1))
    p_F1 = stats.f.sf(F1, q1, n - m3['k'] - 1)
    # Equivalent GLRT chi-square form: -2 log Lambda = n * log(SSR_r/SSR_f)
    glrt1 = n * np.log(m1['ss_res'] / m3['ss_res'])
    p_glrt1 = stats.chi2.sf(glrt1, q1)

    print(f"\n  Nested F-test (M1 -> M3, does draft group add explanatory power?):")
    print(f"    F({q1}, {n-m3['k']-1}) = {F1:.3f}, {fmt_p(p_F1)}")
    print(f"    Equivalent GLRT: -2logΛ = {glrt1:.3f} ~ χ²({q1}), {fmt_p(p_glrt1)}")
    print(f"    {'❌ Reject H0 — draft group adds significant explanatory power beyond rookie-season attributes' if p_F1 < 0.05 else '✅ Fail to reject'}")

    q2 = m3['k'] - m2['k']
    F2 = ((m2['ss_res'] - m3['ss_res']) / q2) / (m3['ss_res'] / (n - m3['k'] - 1))
    p_F2 = stats.f.sf(F2, q2, n - m3['k'] - 1)
    print(f"\n  Nested F-test (M2 -> M3, do rookie-season attributes add power beyond draft group?):")
    print(f"    F({q2}, {n-m3['k']-1}) = {F2:.3f}, {fmt_p(p_F2)}")
    print(f"    {'❌ Reject H0 — rookie-season attributes add significant explanatory power beyond draft group' if p_F2 < 0.05 else '✅ Fail to reject'}")

    print(f"\n  >> Interpretation: both directions {'are' if (p_F1<0.05 and p_F2<0.05) else 'are NOT both'} "
          f"significant, meaning rookie-season attributes and draft group each carry "
          f"independent explanatory power not subsumed by the other.")

    # Bar chart comparing the three models' R²
    fig, ax = plt.subplots(figsize=(7, 5))
    model_names = ["M1: Rookie\nAttributes", "M2: Draft Round\nonly", "M3: Both\ncombined"]
    r2_vals = [m1['r2'], m2['r2'], m3['r2']]
    bars = ax.bar(model_names, r2_vals, color=[ACCENT, GOLD, NAVY],
                  edgecolor="white", width=0.6)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                ha="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.set_ylabel("R²")
    ax.set_title(f"Nested Model Comparison\nM1→M3 F={F1:.1f} ({fmt_p(p_F1)}); "
                 f"M2→M3 F={F2:.1f} ({fmt_p(p_F2)})",
                 color=NAVY, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig("fig17_nested_models.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig17_nested_models.png saved")

    return {"m1": m1, "m2": m2, "m3": m3, "p_F1": p_F1, "p_F2": p_F2}


# ════════════════════════════════════════════════════════════════════
# 7C. REGRESSION WITH INTERACTION TERM  (PPG × Draft Round)
# ════════════════════════════════════════════════════════════════════

def regression_interaction(data):
    print("\n" + "═"*55)
    print("  REGRESSION WITH INTERACTION TERM  (PPG × Draft Round)")
    print("═"*55)

    if "draft_group" not in data.columns:
        print("  ⚠️  'draft_group' not found – skipping")
        return

    sub = data[["PPG", "draft_group", "career_length"]].dropna().copy()
    sub["is_round2"]    = (sub["draft_group"] == "Round 2+").astype(int)
    sub["is_undrafted"] = (sub["draft_group"] == "Undrafted").astype(int)
    ppg_mean, ppg_std = sub["PPG"].mean(), sub["PPG"].std()
    sub["PPG_z"] = (sub["PPG"] - ppg_mean) / ppg_std
    sub["PPG_x_round2"]    = sub["PPG_z"] * sub["is_round2"]
    sub["PPG_x_undrafted"] = sub["PPG_z"] * sub["is_undrafted"]

    feat_cols = ["PPG_z", "is_round2", "is_undrafted",
                 "PPG_x_round2", "PPG_x_undrafted"]
    X = np.column_stack([np.ones(len(sub)), sub[feat_cols].values])
    y = sub["career_length"].values

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    n, k = len(y), X.shape[1] - 1
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    mse = ss_res / (n - k - 1)
    se_b = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * mse))
    t_stat = beta / se_b
    p_vals = np.array([two_sided_t_p(t, n - k - 1) for t in t_stat])

    names = ["const"] + feat_cols
    print(f"\n  R² = {r2:.4f}   n={n}")
    print(f"\n  {'Term':<18} {'β':>8} {'SE':>8} {'t':>8} {'p':>12}")
    print("  " + "-"*58)
    for i, name in enumerate(names):
        sig = " *" if p_vals[i] < 0.05 else ""
        print(f"  {name:<18} {beta[i]:>8.4f} {se_b[i]:>8.4f} "
              f"{t_stat[i]:>8.3f} {fmt_p(p_vals[i]):>12}{sig}")

    p_round2    = p_vals[names.index("PPG_x_round2")]
    p_undrafted = p_vals[names.index("PPG_x_undrafted")]
    print(f"\n  Interaction (PPG × Round2+):    {fmt_p(p_round2)}  "
          f"{'✅ significant' if p_round2 < 0.05 else '— not significant'}")
    print(f"  Interaction (PPG × Undrafted):  {fmt_p(p_undrafted)}  "
          f"{'✅ significant' if p_undrafted < 0.05 else '— NOT significant'}")
    if p_round2 < 0.05 and p_undrafted >= 0.05:
        n_high_ppg_undrafted = ((sub["is_undrafted"]==1) & (sub["PPG_z"] > 1)).sum()
        print(f"  ⚠️  NOTE: the interaction is significant for Round-2+ picks "
              f"but not for Undrafted players. Only {n_high_ppg_undrafted} "
              f"undrafted players have PPG_z > 1 (high scoring) in this "
              f"sample -- this thin cell limits the power to detect an "
              f"interaction for that subgroup specifically, distinct from "
              f"a genuine null effect.")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Round 1": NAVY, "Round 2+": GOLD, "Undrafted": ACCENT}
    for grp in ["Round 1", "Round 2+", "Undrafted"]:
        d = sub[sub["draft_group"] == grp]
        if len(d) < 2:
            continue
        ax.scatter(d["PPG"], d["career_length"], alpha=0.25,
                   color=colors[grp], s=12)
        m, b = np.polyfit(d["PPG"], d["career_length"], 1)
        xs = np.linspace(d["PPG"].min(), d["PPG"].max(), 50)
        ax.plot(xs, m*xs + b, color=colors[grp], linewidth=2.5, label=grp)
    ax.set_xlabel("Points Per Game (rookie season)")
    ax.set_ylabel("Career Length (seasons)")
    ax.set_title(f"PPG Effect on Career Length, by Draft Group\n"
                 f"Round2+ interaction {fmt_p(p_round2)}; "
                 f"Undrafted interaction {fmt_p(p_undrafted)} (n.s.)",
                 color=NAVY, fontweight="bold", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig11_interaction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig11_interaction.png saved")


# ════════════════════════════════════════════════════════════════════
# 7D. CLASSIFICATION  (Logistic Regression: Short vs Long career)
# ════════════════════════════════════════════════════════════════════

def classification_model(data):
    print("\n" + "═"*55)
    print("  CLASSIFICATION – Logistic Regression (Short vs Long career)")
    print("═"*55)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                  roc_curve, roc_auc_score)
    from sklearn.preprocessing import StandardScaler

    # Feature set matches M3 from the nested model comparison (rookie
    # attributes + draft group dummies), not just M1, since H4 showed
    # draft group carries independent explanatory power that a
    # performance-only classifier would be missing.
    base_features = [c for c in ["PPG", "RPG", "APG", "net_rtg", "usg",
                                  "height_cm", "weight_kg", "draft_age"]
                     if c in data.columns]
    use_draft = "draft_group" in data.columns
    cols_needed = base_features + (["draft_group"] if use_draft else []) + ["career_length"]
    sub = data[cols_needed].dropna().copy()
    if use_draft:
        sub["is_round2"] = (sub["draft_group"] == "Round 2+").astype(int)
        sub["is_undrafted"] = (sub["draft_group"] == "Undrafted").astype(int)
    features = base_features + (["is_round2", "is_undrafted"] if use_draft else [])

    threshold = sub["career_length"].median()
    sub["long_career"] = (sub["career_length"] >= threshold).astype(int)
    majority_class_pct = max(sub["long_career"].mean(), 1 - sub["long_career"].mean())
    print(f"  Features: {features}")
    print(f"  Threshold: career_length ≥ {threshold:.0f} seasons = 'Long'")
    print(f"  Class balance: {sub['long_career'].mean():.1%} Long, "
          f"{1-sub['long_career'].mean():.1%} Short")
    print(f"  Majority-class baseline accuracy: {majority_class_pct:.1%}  "
          f"(this — not 50% — is the correct comparison baseline)")

    X = sub[features].values
    y = sub["long_career"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  Test Accuracy: {acc:.3f}  (vs. {majority_class_pct:.3f} majority-class baseline)")
    print(f"  AUC:           {auc:.3f}")
    print("\n  Confusion Matrix:")
    print(f"                 Pred Short  Pred Long")
    print(f"  Actual Short   {cm[0,0]:>10}  {cm[0,1]:>9}")
    print(f"  Actual Long    {cm[1,0]:>10}  {cm[1,1]:>9}")

    print("\n  Coefficients (odds-ratio direction):")
    for feat, coef in zip(features, clf.coef_[0]):
        direction = "↑ increases" if coef > 0 else "↓ decreases"
        print(f"    {feat:<12}: {coef:>7.3f}  ({direction} odds of long career)")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=NAVY, linewidth=2.5, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GOLD, linewidth=1.5, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Long Career Classification",
                 color=NAVY, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig12_roc.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Short", "Long"], yticklabels=["Short", "Long"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Accuracy={acc:.3f}, baseline={majority_class_pct:.3f})",
                 color=NAVY, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig("fig13_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig12_roc.png, fig13_confusion.png saved")
    return {"accuracy": acc, "auc": auc, "baseline": majority_class_pct}


# ════════════════════════════════════════════════════════════════════
# 7E. CROSS-VALIDATION  (validate the regression model)
# ════════════════════════════════════════════════════════════════════

def cross_validation(data):
    print("\n" + "═"*55)
    print("  CROSS-VALIDATION – Regression Model Robustness")
    print("═"*55)

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Validate M3 (rookie attributes + draft group), the paper's actual
    # headline regression model -- not M1 alone, which would validate a
    # weaker model than the one the paper's conclusions rest on.
    base_features = [c for c in ["PPG", "RPG", "APG", "net_rtg", "usg",
                                  "height_cm", "weight_kg", "draft_age"]
                     if c in data.columns]
    use_draft = "draft_group" in data.columns
    cols_needed = base_features + (["draft_group"] if use_draft else []) + ["career_length"]
    sub = data[cols_needed].dropna().copy()
    if use_draft:
        sub["is_round2"] = (sub["draft_group"] == "Round 2+").astype(int)
        sub["is_undrafted"] = (sub["draft_group"] == "Undrafted").astype(int)
    features = base_features + (["is_round2", "is_undrafted"] if use_draft else [])
    print(f"  Features (M3 -- rookie attributes + draft group): {features}")

    X = sub[features].values
    y = sub["career_length"].values

    # Scale inside a Pipeline so each fold's scaler is fit only on that
    # fold's training data (avoids any leakage from the held-out fold
    # into the feature scaling). For plain OLS this doesn't change R^2
    # -- linear regression is invariant to per-column affine rescaling
    # -- but this is the correct general pattern to follow.
    pipe = Pipeline([("scale", StandardScaler()), ("ols", LinearRegression())])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    pipe.fit(X, y)
    insample_r2 = pipe.score(X, y)

    print(f"\n  In-sample R² (fit & score on same data): {insample_r2:.4f}")
    print(f"  5-Fold CV R² per fold: {np.round(cv_scores, 4)}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f}  (± {cv_scores.std():.4f})")
    gap = insample_r2 - cv_scores.mean()
    print(f"\n  Overfitting gap: {gap:.4f}")
    print(f"  {'⚠️  Some overfitting detected' if gap > 0.03 else '✅ Model generalizes well — minimal overfitting'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(1, 6), cv_scores, color=NAVY, edgecolor=GOLD, width=0.6)
    ax.axhline(cv_scores.mean(), color=GOLD, linewidth=2, linestyle="--",
              label=f"Mean CV R²={cv_scores.mean():.3f}")
    ax.axhline(insample_r2, color="#B22222", linewidth=2, linestyle=":",
              label=f"In-sample R²={insample_r2:.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("R²")
    ax.set_title("5-Fold Cross-Validation — M3 (Rookie Attributes + Draft) Stability",
                 color=NAVY, fontweight="bold", fontsize=11)
    ax.set_xticks(range(1, 6))
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig14_crossval.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig14_crossval.png saved")


# ════════════════════════════════════════════════════════════════════
# 8. CHI-SQUARE  (מבחן אי-תלות)
# ════════════════════════════════════════════════════════════════════

def chi_square_test(data):
    print("\n" + "═"*55)
    print("  CHI-SQUARE – Draft Group × Career Category")
    print("═"*55)
    if "draft_group" not in data.columns:
        print("  ⚠️  'draft_group' not found – skipping")
        return

    d = data.copy()
    d["career_cat"] = pd.cut(d["career_length"],
                             bins=[0, 3, 7, 100],
                             labels=["Short (≤3)", "Medium (4–7)", "Long (8+)"])
    ct = pd.crosstab(d["draft_group"], d["career_cat"])
    print("\n  Contingency Table:")
    print(ct)

    chi2_stat, p, dof, _ = chi2_contingency(ct)
    print(f"\n  χ²={chi2_stat:.3f}, df={dof}, {fmt_p(p)}")
    print(f"  {'❌ Reject H0 – dependent' if p<0.05 else '✅ Fail to reject – independent'}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", ax=ax, color=[NAVY, GOLD, ACCENT],
                edgecolor="white", width=0.7)
    ax.set_xlabel("Draft Group")
    ax.set_ylabel("Proportion (%)")
    ax.set_title(f"Career Category by Draft Group\nχ²={chi2_stat:.2f}, {fmt_p(p)}",
                 color=NAVY, fontweight="bold")
    ax.legend(title="Career Length")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("fig8_chisquare.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig8_chisquare.png saved")


# ════════════════════════════════════════════════════════════════════
# 9. WALD SEQUENTIAL TEST  (זמני עצירה – SPRT)
#
#    The per-observation log-likelihood-ratio increment for comparing
#    N(mu1,sigma) against N(mu0,sigma) is
#        logLR(x) = (mu1-mu0)/sigma^2 * x  -  (mu1^2-mu0^2)/(2*sigma^2)
#    Because a single sequence's stopping point depends heavily on the
#    order in which observations arrive, we run the test across 1000
#    random re-orderings and report the full distribution of decisions
#    and stopping times, together with the empirical Average Sample
#    Number (ASN) compared against Wald's theoretical approximation.
# ════════════════════════════════════════════════════════════════════

def wald_sequential_test(data, alpha=0.05, beta=0.20, n_sims=1000):
    print("\n" + "═"*55)
    print("  WALD SEQUENTIAL TEST (SPRT) – 1000-seed simulation")
    print("═"*55)

    base_vals = data[data["PPG"] >= data["PPG"].median()]["career_length"].dropna().values
    mu0   = data["career_length"].mean()
    sigma = data["career_length"].std()
    mu1   = mu0 + 0.3 * sigma
    true_mean_subgroup = base_vals.mean()

    A = np.log((1 - beta) / alpha)    # Reject H0 boundary
    B = np.log(beta / (1 - alpha))    # Accept H0 boundary

    print(f"  μ₀={mu0:.2f} (H0), μ₁={mu1:.2f} (H1), σ={sigma:.2f}")
    print(f"  True subgroup mean = {true_mean_subgroup:.2f} "
          f"({'closer to H1' if abs(true_mean_subgroup-mu1) < abs(true_mean_subgroup-mu0) else 'closer to H0'})")
    print(f"  Boundaries: A={A:.3f} (reject H0), B={B:.3f} (accept H0)")

    def sprt_run(vals):
        llr, path = 0.0, [0.0]
        for i, x in enumerate(vals):
            # CORRECTED log-likelihood-ratio increment
            llr += (mu1 - mu0) / sigma**2 * x - (mu1**2 - mu0**2) / (2 * sigma**2)
            path.append(llr)
            if llr >= A:
                return i + 1, "Reject H0", path
            elif llr <= B:
                return i + 1, "Accept H0", path
        return None, "No decision", path

    decisions, stops = [], []
    for seed in range(n_sims):
        vals = base_vals.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(vals)
        stop_n, decision, path = sprt_run(vals)
        decisions.append(decision)
        if stop_n is not None:
            stops.append(stop_n)

    decisions = np.array(decisions)
    pct_reject = (decisions == "Reject H0").mean()
    pct_accept = (decisions == "Accept H0").mean()
    pct_none   = (decisions == "No decision").mean()
    stops = np.array(stops)

    print(f"\n  Across {n_sims} random orderings of the high-PPG subgroup:")
    print(f"    Reject H0 (supports H1): {pct_reject:.1%}")
    print(f"    Accept H0:               {pct_accept:.1%}")
    print(f"    No decision reached:     {pct_none:.1%}")
    if len(stops) > 0:
        print(f"    Empirical ASN (avg stopping n, decided runs): {stops.mean():.1f} "
              f"(median={np.median(stops):.0f}, min={stops.min()}, max={stops.max()})")

    # Wald's theoretical ASN approximation under H1:
    # E[Z|H1] = KL(f1||f0) for these two normals with equal variance
    EZ_H1 = (mu1 - mu0)**2 / (2 * sigma**2)
    ASN_H1_theory = abs(((1 - beta) * A + beta * B) / EZ_H1)
    print(f"    Theoretical Wald ASN (under H1 assumption): ≈{ASN_H1_theory:.1f}")

    # Plot 1: distribution of stopping times
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(stops, bins=30, color=NAVY, edgecolor=GOLD)
    axes[0].axvline(stops.mean(), color=GOLD, linewidth=2, linestyle="--",
                    label=f"Mean n={stops.mean():.1f}")
    axes[0].set_xlabel("Stopping Observation Number (n)")
    axes[0].set_ylabel("Count (out of {} runs)".format(n_sims))
    axes[0].set_title("Distribution of SPRT Stopping Times", color=NAVY, fontweight="bold")
    axes[0].legend()

    labels = ["Reject H0\n(supports H1)", "Accept H0", "No decision"]
    vals_bar = [pct_reject, pct_accept, pct_none]
    axes[1].bar(labels, vals_bar, color=[NAVY, ACCENT, "#AAAAAA"], edgecolor="white")
    for i, v in enumerate(vals_bar):
        axes[1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")
    axes[1].set_ylabel("Proportion of 1000 simulations")
    axes[1].set_title("SPRT Decision Distribution", color=NAVY, fontweight="bold")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("fig9_wald.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig9_wald.png saved (now shows 1000-simulation distribution, not a single run)")

    return {"pct_reject": pct_reject, "pct_accept": pct_accept,
            "mean_stop": stops.mean() if len(stops) else np.nan}


# ════════════════════════════════════════════════════════════════════
# 10. CORRELATION MATRIX
# ════════════════════════════════════════════════════════════════════

def correlation_matrix(data):
    print("\n" + "═"*55)
    print("  CORRELATION MATRIX")
    print("═"*55)
    cols = [c for c in ["career_length", "PPG", "RPG", "APG",
                         "net_rtg", "usg", "height_cm", "draft_age"]
            if c in data.columns]
    corr = data[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix – Key Variables",
                 color=NAVY, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig("fig10_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → fig10_correlation.png saved")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🏀  NBA Career Longevity – Full Statistical Analysis\n")
    print("  ⚠️  Make sure 'all_seasons.csv' is in the same folder!")
    print("  Download: https://www.kaggle.com/datasets/justinas/nba-players-data\n")

    data = load_data("all_seasons.csv")

    descriptive_stats(data)
    goodness_of_fit(data)
    hypothesis_ppg(data)
    hypothesis_draft_round(data)
    hypothesis_age(data)
    regression(data)
    nested_model_comparison(data)
    regression_interaction(data)
    classification_model(data)
    cross_validation(data)
    chi_square_test(data)
    wald_sequential_test(data)
    correlation_matrix(data)

    print("\n✅  Done! Figures saved as fig1–fig18 PNG files.")
    print("    Use them directly in your Phase 2 paper.\n")
