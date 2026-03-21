"""
season_simulator.py

Monte Carlo season simulation for fantasy baseball standings prediction.

Data sources (all from fantasy_data.db)
─────────────────────────────────────────
  players{year}  — mean projections, z-scores, InterSD, percentiles, Owner
  eligibility    — all_pos (position eligibility) by year/week
  players{yr-1}  — Age column (not present in current year's table)

Architecture
─────────────
  load_simulation_data()   → builds the merged DataFrame the sim needs
  PlayerVarianceModel      → sampling distributions per player from InterSD + age
  InjuryModel              → weekly availability masks
  SeasonSimulator          → week-by-week Monte Carlo loop
  score_roto()             → ranks teams across 10 categories → roto points

Roto categories
───────────────
  Hitting  : R, HR, RBI, SB, BA  (BA derived from H/AB)
  Pitching : QS, SO, SvHld, ERA, WHIP  (ERA from ER/IP*9, WHIP from (HA+BB)/IP)
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimize_lineup import Optimized_Lineups

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_WEEKS = 26
ROTO_CATS_H = ['R', 'HR', 'RBI', 'SB', 'BA']
ROTO_CATS_P = ['QS', 'SO', 'SvHld', 'ERA', 'WHIP']
ROTO_CATS_LOWER_IS_BETTER = {'ERA', 'WHIP'}

# Counting stat components we sample (rate stats derived from these at scoring)
HITTER_COUNT_STATS = ['PA', 'H', 'AB', 'R', 'HR', 'RBI', 'SB']
PITCHER_COUNT_STATS = ['IP', 'ER', 'HA', 'BB', 'SO', 'QS', 'SvHld']

# Positions that identify a hitter vs pitcher
HITTER_POSITIONS = {'C', '1B', '2B', '3B', 'OF', 'DH', 'SS'}

# Minimum cross-system std as fraction of mean (floor for converged projections)
MIN_STD_FRACTION = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# Variance model
# ─────────────────────────────────────────────────────────────────────────────

# Age-based coefficient-of-variation multiplier applied on top of InterSD.
# The projection systems converge tightly on unproven players even though
# their true outcome distribution is much wider (breakout / bust / IP limits).
# These values are starting-point estimates — calibrate against historical
# projection error data when available.
AGE_CV_MULTIPLIERS: List[Tuple[int, int, float]] = [
    (0,  23, 3.5),
    (24, 26, 2.5),
    (27, 29, 1.5),
    (30, 34, 1.0),
    (35, 99, 1.2),  # age-related decline risk is real
]

DEFAULT_CV_BY_POSITION = {
    'C':  0.28,
    'SS': 0.25,
    '2B': 0.25,
    '3B': 0.25,
    'OF': 0.25,
    '1B': 0.22,
    'DH': 0.22,
    'SP': 0.35,   # IP uncertainty dominates
    'RP': 0.40,   # role/saves-opp uncertainty is high
    'P':  0.38,
}
DEFAULT_CV_FALLBACK = 0.30


def _age_multiplier(age: float) -> float:
    for lo, hi, mult in AGE_CV_MULTIPLIERS:
        if lo <= age <= hi:
            return mult
    return 1.0


@dataclass
class PlayerDistribution:
    cbsid: int
    name: str
    player_type: str   # 'h' or 'p'
    age: float
    all_pos: list
    means: Dict[str, float]
    stds:  Dict[str, float]


class PlayerVarianceModel:
    """
    Builds per-player sampling distributions from the players{year} table.

    Variance source
    ───────────────
    InterSD is the inter-system standard deviation of the player's total
    z-score across projection systems. We use it as a coefficient of
    variation (CV = InterSD / |z|) and apply that CV to each counting stat.

    For players without InterSD (new/unproven), we fall back to a
    position-based default CV.

    Age adjustment
    ──────────────
    For players ≤26, systems converge methodologically even though real
    outcome uncertainty is much higher (breakouts, role changes, IP limits).
    We amplify the CV by AGE_CV_MULTIPLIERS to model this.

    Volume/rate separation
    ──────────────────────
    PA (hitters) and IP (pitchers) are sampled first as the primary volume
    driver. All counting stats are then scaled proportionally, keeping
    per-PA/per-IP rates stable while allowing volume uncertainty to produce
    Crochet-style IP boom-or-bust outcomes.
    """

    def __init__(self, players_df: pd.DataFrame):
        self._dists: Dict[int, PlayerDistribution] = {}
        self._build(players_df)

    def _build(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            cbsid = int(row['cbsid'])
            ptype = str(row.get('type', 'h'))
            age   = float(row.get('Age') or 28.0)
            name  = str(row.get('Player', row.get('Name', str(cbsid))))
            all_pos = row.get('all_pos', [])
            if not isinstance(all_pos, list):
                all_pos = []

            # ── Coefficient of variation ──────────────────────────────────
            inter_sd = row.get('InterSD')
            z        = float(row.get('z') or 0.0)
            if inter_sd is not None and not pd.isna(inter_sd) and z != 0:
                cv = abs(float(inter_sd) / z)
            else:
                primary_pos = str(row.get('Primary_Pos', ''))
                cv = DEFAULT_CV_BY_POSITION.get(primary_pos, DEFAULT_CV_FALLBACK)

            cv = cv * _age_multiplier(age)
            cv = max(cv, MIN_STD_FRACTION)    # floor
            cv = min(cv, 2.0)                 # cap against extreme outliers

            # ── Per-stat distributions ────────────────────────────────────
            stat_cols = HITTER_COUNT_STATS if ptype == 'h' else PITCHER_COUNT_STATS
            means: Dict[str, float] = {}
            stds:  Dict[str, float] = {}
            for stat in stat_cols:
                raw = row.get(stat)
                if raw is not None and not pd.isna(raw):
                    mu = float(raw)
                    means[stat] = mu
                    stds[stat]  = max(abs(mu * cv), abs(mu) * MIN_STD_FRACTION, 0.01)

            self._dists[cbsid] = PlayerDistribution(
                cbsid=cbsid,
                name=name,
                player_type=ptype,
                age=age,
                all_pos=all_pos,
                means=means,
                stds=stds,
            )

    def get_distribution(self, cbsid: int) -> Optional[PlayerDistribution]:
        return self._dists.get(cbsid)

    def sample(self, dist: PlayerDistribution, rng: np.random.Generator) -> Dict[str, float]:
        """
        Sample one season stat line.

        Volume (PA or IP) is drawn first; all counting stats are then
        scaled by the volume ratio so that per-PA/per-IP rates stay coherent
        while overall output varies. This is the key mechanism for modeling
        IP-limited pitchers (the Crochet scenario).
        """
        result: Dict[str, float] = {}

        if dist.player_type == 'h':
            pa_mu  = dist.means.get('PA', 500)
            pa_sig = dist.stds.get('PA',  80)
            pa_sampled = max(0.0, rng.normal(pa_mu, pa_sig))
            vol_scale  = pa_sampled / pa_mu if pa_mu > 0 else 0.0
            result['PA'] = pa_sampled
            result['AB'] = max(0.0, dist.means.get('AB', pa_mu * 0.88) * vol_scale)
            for stat in ['H', 'R', 'HR', 'RBI', 'SB']:
                if stat in dist.means:
                    result[stat] = max(0.0, dist.means[stat] * vol_scale +
                                       rng.normal(0, dist.stds[stat] * 0.3))
        else:
            ip_mu  = dist.means.get('IP', 60)
            ip_sig = dist.stds.get('IP', 20)
            ip_sampled = max(0.0, rng.normal(ip_mu, ip_sig))
            vol_scale  = ip_sampled / ip_mu if ip_mu > 0 else 0.0
            result['IP'] = ip_sampled
            for stat in ['ER', 'HA', 'BB', 'SO', 'QS', 'SvHld']:
                if stat in dist.means:
                    result[stat] = max(0.0, dist.means[stat] * vol_scale +
                                       rng.normal(0, dist.stds[stat] * 0.3))

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Injury model
# ─────────────────────────────────────────────────────────────────────────────

# (prob_any_stint, mean_weeks_out, std_weeks_out)
# These are rough calibration values. Calibrate against historical MLB IL data.
IL_PARAMS: Dict[str, Tuple[float, float, float]] = {
    'C':  (0.45, 4.0, 2.5),
    'SS': (0.30, 3.0, 2.0),
    '2B': (0.30, 3.0, 2.0),
    '3B': (0.30, 3.0, 2.0),
    'OF': (0.30, 3.5, 2.0),
    '1B': (0.28, 3.0, 2.0),
    'DH': (0.28, 3.0, 2.0),
    'SP': (0.40, 5.0, 4.0),  # wide std: TJ vs blister are very different
    'RP': (0.35, 3.0, 2.0),
    'P':  (0.38, 4.0, 3.0),
}
IL_DEFAULT = (0.30, 3.0, 2.0)


class InjuryModel:
    """
    Samples per-player IL availability as a boolean mask of length n_weeks.

    One IL stint per player per simulation (extend to two stints via TODO).
    Start week is uniform; duration is drawn from a normal distribution
    truncated at 1 week minimum — players come back from one injury rather
    than re-injuring week over week.
    """

    def __init__(self, n_weeks: int = N_WEEKS):
        self.n_weeks = n_weeks

    def sample_availability(
        self,
        primary_pos: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask = np.ones(self.n_weeks, dtype=bool)
        prob, mean_wks, std_wks = IL_PARAMS.get(primary_pos, IL_DEFAULT)
        if rng.random() < prob:
            start = int(rng.integers(0, self.n_weeks))
            dur   = max(1, int(round(rng.normal(mean_wks, std_wks))))
            mask[start : min(self.n_weeks, start + dur)] = False
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Roto scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_roto(
    team_season_stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Rank n teams in each roto category and sum to roto points.

    Input stats expected per team:
      Counting : R, HR, RBI, SB, H, AB, QS, SO, SvHld
      Components: ER, IP, HA, BB  (ERA and WHIP derived from these)

    Returns a DataFrame indexed by owner, sorted by total roto points.
    """
    rows: Dict[str, dict] = {}
    for owner, stats in team_season_stats.items():
        ip = max(stats.get('IP', 0.0), 0.01)
        ab = max(stats.get('AB', 0.0), 0.01)
        rows[owner] = {
            'R':      stats.get('R',      0.0),
            'HR':     stats.get('HR',     0.0),
            'RBI':    stats.get('RBI',    0.0),
            'SB':     stats.get('SB',     0.0),
            'BA':     stats.get('H', 0.0) / ab,
            'QS':     stats.get('QS',     0.0),
            'SO':     stats.get('SO',     0.0),
            'SvHld':  stats.get('SvHld',  0.0),
            'ERA':    stats.get('ER', 0.0) * 9.0 / ip,
            'WHIP':   (stats.get('HA', 0.0) + stats.get('BB', 0.0)) / ip,
        }

    stat_df = pd.DataFrame(rows).T
    all_cats = ROTO_CATS_H + ROTO_CATS_P

    scores = pd.DataFrame(index=list(team_season_stats.keys()))
    for cat in all_cats:
        ascending = cat in ROTO_CATS_LOWER_IS_BETTER
        scores[cat] = stat_df[cat].rank(ascending=ascending, method='average')

    scores['total'] = scores[all_cats].sum(axis=1)
    return scores.sort_values('total', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Season simulator
# ─────────────────────────────────────────────────────────────────────────────

class SeasonSimulator:
    """
    Monte Carlo simulation of a full fantasy baseball season.

    Parameters
    ──────────
    rosters        : {owner: DataFrame}
                     Each DataFrame is one team's full roster in the format
                     returned by load_simulation_data() — must contain:
                       cbsid, Player, all_pos, type, Primary_Pos, Owner, z
                     plus the counting stat columns used by PlayerVarianceModel.
    variance_model : PlayerVarianceModel built from the same data
    n_sims         : Monte Carlo iterations (1 000 is a reasonable start)
    n_weeks        : Scoring weeks in the season
    optimize_col   : Column passed to Optimized_Lineups (default 'z')

    Usage
    ─────
    data            = load_simulation_data(engine, year=2026)
    rosters         = build_rosters(data)
    variance_model  = PlayerVarianceModel(data)
    sim             = SeasonSimulator(rosters, variance_model, n_sims=1000)
    summary         = sim.run()
    print(summary[['owner','pts_mean','pts_median','p_1st','p_top3','p_top6']])
    """

    def __init__(
        self,
        rosters: Dict[str, pd.DataFrame],
        variance_model: PlayerVarianceModel,
        n_sims: int = 1_000,
        n_weeks: int = N_WEEKS,
        optimize_col: str = 'z',
    ):
        self.rosters        = rosters
        self.variance_model = variance_model
        self.injury_model   = InjuryModel(n_weeks)
        self.n_sims         = n_sims
        self.n_weeks        = n_weeks
        self.optimize_col   = optimize_col
        self.sim_records:   List[pd.DataFrame] = []

        # Index every rostered player for fast lookup
        self._player_meta: Dict[int, Dict] = {}
        for df in rosters.values():
            for _, row in df.iterrows():
                cid = int(row['cbsid'])
                if cid not in self._player_meta:
                    self._player_meta[cid] = {
                        'primary_pos': str(row.get('Primary_Pos', '')),
                    }

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, seed: int = 42) -> pd.DataFrame:
        """Run all simulations; return summary DataFrame."""
        rng_master = np.random.default_rng(seed)
        self.sim_records = []

        for i in range(self.n_sims):
            child_seed = int(rng_master.integers(0, 2**31))
            standings  = self._run_one(np.random.default_rng(child_seed))
            self.sim_records.append(standings)
            if (i + 1) % 100 == 0:
                logger.info(f"Sim {i+1}/{self.n_sims} done")

        return self._summarize()

    # ── Single simulation ─────────────────────────────────────────────────────

    def _run_one(self, rng: np.random.Generator) -> pd.DataFrame:
        # 1. Sample season stat lines for all rostered players
        player_samples: Dict[int, Dict[str, float]] = {}
        for cid, meta in self._player_meta.items():
            dist = self.variance_model.get_distribution(cid)
            player_samples[cid] = (
                self.variance_model.sample(dist, rng) if dist else {}
            )

        # 2. Sample injury availability masks
        availability: Dict[int, np.ndarray] = {
            cid: self.injury_model.sample_availability(
                meta['primary_pos'], rng
            )
            for cid, meta in self._player_meta.items()
        }

        # 3. Week-by-week loop
        team_totals: Dict[str, Dict[str, float]] = {
            owner: defaultdict(float) for owner in self.rosters
        }
        lineup_cache: Dict[str, Tuple] = {}

        for week in range(self.n_weeks):
            for owner, full_roster in self.rosters.items():
                healthy_df = self._healthy_roster(full_roster, availability, week)
                lineup     = self._get_lineup(owner, healthy_df, lineup_cache)
                self._accumulate_week(team_totals[owner], lineup, player_samples)

        # 4. Score roto
        return score_roto(team_totals)

    # ── Weekly helpers ────────────────────────────────────────────────────────

    def _healthy_roster(
        self,
        full_roster: pd.DataFrame,
        availability: Dict[int, np.ndarray],
        week: int,
    ) -> pd.DataFrame:
        mask = full_roster['cbsid'].apply(
            lambda c: bool(availability.get(int(c),
                           np.ones(self.n_weeks, bool))[week])
        )
        return full_roster[mask].copy()

    def _get_lineup(
        self,
        owner: str,
        healthy_df: pd.DataFrame,
        cache: Dict,
    ) -> List[int]:
        """
        Return cbsids for the optimal active lineup from healthy_df.

        Uses Optimized_Lineups ILP; caches results keyed by the frozenset of
        healthy cbsids so identical rosters skip the solver.
        """
        if healthy_df.empty:
            return []

        healthy_key = frozenset(healthy_df['cbsid'].astype(int).tolist())
        cache_key   = f"{owner}_{hash(healthy_key)}"

        if cache_key in cache and cache[cache_key][0] == healthy_key:
            return cache[cache_key][1]

        # Tag all rows with this owner so Optimized_Lineups can filter internally.
        # Deduplicate on Player name (keep highest z) to avoid set_index failures.
        temp_df = (
            healthy_df.copy()
            .sort_values(self.optimize_col, ascending=False)
            .drop_duplicates(subset='Player', keep='first')
            .reset_index(drop=True)
        )
        temp_df['Owner'] = owner

        # Build name→cbsid map before optimization (guaranteed unique names now)
        name_to_cbsid: Dict[str, int] = dict(
            zip(temp_df['Player'], temp_df['cbsid'].astype(int))
        )

        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter('ignore')   # suppress per-week ILP infeasible noise
                opt = Optimized_Lineups(
                    owner=owner,
                    data=temp_df,
                    optimize_col=self.optimize_col,
                    player_col='Player',
                    pos_col='all_pos',
                    owner_col='Owner',
                    type_col='type',
                )
                opt.optimize()

            active_names = (
                list(opt.hitter_optimized_lineup.values())
                + list(opt.pitcher_optimized_lineup.values())
            )
            lineup_cbsids = [
                name_to_cbsid[n] for n in active_names if n in name_to_cbsid
            ]

            # ILP may return empty/partial hitter lineup without raising
            # (e.g. no catcher on roster). Fall back to greedy in that case.
            type_map = dict(zip(temp_df['Player'], temp_df['type']))
            n_hitters = sum(1 for n in active_names if type_map.get(n) == 'h')
            if n_hitters < 10:
                lineup_cbsids = self._greedy_lineup(temp_df, name_to_cbsid)

        except Exception:
            lineup_cbsids = self._greedy_lineup(temp_df, name_to_cbsid)

        cache[cache_key] = (healthy_key, lineup_cbsids)
        return lineup_cbsids

    def _greedy_lineup(
        self,
        temp_df: pd.DataFrame,
        name_to_cbsid: Dict[str, int],
    ) -> List[int]:
        """
        Fallback when ILP is infeasible (e.g. no catcher on roster).
        Takes the top-14 hitters and top-9 pitchers by optimize_col.
        No positional constraints — just best available by value.
        """
        col = self.optimize_col
        hitters  = temp_df[temp_df['type'] == 'h'].nlargest(14, col)
        pitchers = temp_df[temp_df['type'] == 'p'].nlargest(9,  col)
        names    = list(hitters['Player']) + list(pitchers['Player'])
        return [name_to_cbsid[n] for n in names if n in name_to_cbsid]

    def _accumulate_week(
        self,
        totals: Dict[str, float],
        lineup_cbsids: List[int],
        player_samples: Dict[int, Dict[str, float]],
    ) -> None:
        """Add one week's share (1/n_weeks) of each active player's season sample."""
        all_stats = HITTER_COUNT_STATS + PITCHER_COUNT_STATS
        for cid in lineup_cbsids:
            sample = player_samples.get(cid, {})
            for stat in all_stats:
                if stat in sample:
                    totals[stat] += sample[stat] / self.n_weeks

    # ── Summarize across simulations ──────────────────────────────────────────

    def _summarize(self) -> pd.DataFrame:
        if not self.sim_records:
            return pd.DataFrame()

        owners   = list(self.rosters.keys())
        all_cats = ROTO_CATS_H + ROTO_CATS_P

        totals_by_owner: Dict[str, List[float]]       = defaultdict(list)
        rank_counts:     Dict[str, Dict[int, int]]    = {o: defaultdict(int) for o in owners}
        cat_pts:         Dict[str, Dict[str, List]]   = {o: defaultdict(list) for o in owners}

        for standings in self.sim_records:
            for owner in owners:
                if owner not in standings.index:
                    continue
                pts  = float(standings.loc[owner, 'total'])
                rank = int((standings['total'] > pts).sum()) + 1
                totals_by_owner[owner].append(pts)
                rank_counts[owner][rank] += 1
                for cat in all_cats:
                    if cat in standings.columns:
                        cat_pts[owner][cat].append(float(standings.loc[owner, cat]))

        rows = []
        for owner in owners:
            arr = np.array(totals_by_owner[owner])
            n   = max(len(arr), 1)
            row = {
                'owner':      owner,
                'pts_mean':   arr.mean(),
                'pts_median': float(np.median(arr)),
                'pts_std':    arr.std(),
                'p_1st':      rank_counts[owner][1] / n,
                'p_top3':     sum(rank_counts[owner][r] for r in range(1, 4)) / n,
                'p_top6':     sum(rank_counts[owner][r] for r in range(1, 7)) / n,
            }
            for cat in all_cats:
                vals = cat_pts[owner].get(cat, [])
                row[f'{cat}_mean'] = float(np.mean(vals)) if vals else 0.0
            rows.append(row)

        summary = pd.DataFrame(rows).sort_values('pts_mean', ascending=False)
        summary.insert(0, 'proj_rank', range(1, len(summary) + 1))
        return summary.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_simulation_data(
    db_path: str,
    year: int,
    eligibility_week: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build the merged DataFrame the simulator needs from the fantasy DB.

    Joins:
      players{year}   — projections, z-scores, InterSD, Owner, Paid
      eligibility     — all_pos for each player (uses latest available week)
      players{yr-1}   — Age column (not present in current year's table)

    Returns a DataFrame with one row per player containing:
      cbsid, Player, all_pos, type, Primary_Pos, Age, Owner, Paid, z, InterSD
      + all counting stat columns (PA, H, AB, R, HR, RBI, SB,
                                   IP, ER, HA, BB, SO, QS, SvHld)

    Notes
    ─────
    - 'type' is 'h' for hitters, 'p' for pitchers (derived from Primary_Pos)
    - 'Player' is an alias for 'Name' (required by Optimized_Lineups)
    - 'all_pos' comes from the eligibility table; falls back to parsing
      the 'Pos' column if no eligibility row is found
    """
    import sqlite3
    conn = sqlite3.connect(db_path)

    # ── 1. Load main projection table ──────────────────────────────────────
    proj = pd.read_sql(f'SELECT * FROM players{year}', conn)
    proj['cbsid'] = pd.to_numeric(proj['cbsid'], errors='coerce')
    proj = proj[proj['cbsid'].notna()].copy()
    proj['cbsid'] = proj['cbsid'].astype(int)

    # Rename Name → Player for Optimized_Lineups compatibility
    if 'Name' in proj.columns and 'Player' not in proj.columns:
        proj = proj.rename(columns={'Name': 'Player'})

    # Derive hitter/pitcher type from Primary_Pos
    proj['type'] = proj['Primary_Pos'].apply(
        lambda p: 'h' if str(p) in HITTER_POSITIONS else 'p'
    )

    # ── 2. Load eligibility (all_pos) ──────────────────────────────────────
    # Use the requested week, or fall back to the latest available
    elig = _load_eligibility(conn, year, eligibility_week)
    proj = proj.merge(
        elig[['cbsid', 'all_pos']],
        on='cbsid',
        how='left',
    )
    # For players not in eligibility, parse Pos column as fallback.
    # Keep as string repr so the column dtype stays consistent.
    missing_elig = proj['all_pos'].isna()
    if missing_elig.any():
        proj['all_pos'] = proj['all_pos'].astype(object)
        proj.loc[missing_elig, 'all_pos'] = proj.loc[missing_elig, 'Pos'].apply(
            lambda p: str(_parse_pos_fallback(p))
        )

    # ── 3. Add Age from prior year's table ─────────────────────────────────
    if 'Age' not in proj.columns or proj['Age'].isna().all():
        age_df = _load_age(conn, year)
        if age_df is not None:
            proj = proj.merge(age_df, on='cbsid', how='left')
            if 'Age_y' in proj.columns:
                proj['Age'] = proj['Age'].fillna(proj['Age_y'])
                proj.drop(columns=['Age_y'], inplace=True)
        if 'Age' not in proj.columns:
            proj['Age'] = 28.0   # league-average fallback
    proj['Age'] = proj['Age'].fillna(28.0)

    # ── 4. Ensure SvHld column ─────────────────────────────────────────────
    if 'SvHld' not in proj.columns:
        sv  = proj.get('SV',  pd.Series(0.0, index=proj.index))
        hld = proj.get('HLD', pd.Series(0.0, index=proj.index))
        proj['SvHld'] = sv.fillna(0) + hld.fillna(0)

    conn.close()
    logger.info(
        f"Loaded {len(proj)} players for {year} "
        f"({proj['Owner'].notna().sum()} owned)"
    )
    return proj


def _load_eligibility(conn, year: int, week: Optional[int]) -> pd.DataFrame:
    """Return eligibility rows for the closest available year/week."""
    if week is not None:
        elig = pd.read_sql(
            f"SELECT cbsid, all_pos FROM eligibility WHERE year={year} AND week={week}",
            conn,
        )
        if not elig.empty:
            return elig

    # Fall back: latest week in the requested year, or latest year overall
    latest = pd.read_sql(
        f"""SELECT cbsid, all_pos FROM eligibility
            WHERE year={year} AND week=(
                SELECT MAX(week) FROM eligibility WHERE year={year}
            )""",
        conn,
    )
    if not latest.empty:
        logger.info(f"Using latest {year} eligibility data")
        return latest

    # Try the prior year as a final fallback
    fallback = pd.read_sql(
        f"""SELECT cbsid, all_pos FROM eligibility
            WHERE year={year-1} AND week=(
                SELECT MAX(week) FROM eligibility WHERE year={year-1}
            )""",
        conn,
    )
    if not fallback.empty:
        logger.warning(
            f"No {year} eligibility data; using {year-1} as fallback. "
            "Positions may be slightly stale."
        )
    return fallback


def _load_age(conn, year: int) -> Optional[pd.DataFrame]:
    """Load Age from the prior year's player table (age + 1)."""
    try:
        age_df = pd.read_sql(
            f"SELECT cbsid, Age FROM players{year-1} WHERE Age IS NOT NULL",
            conn,
        )
        age_df['cbsid'] = pd.to_numeric(age_df['cbsid'], errors='coerce').astype(int)
        age_df['Age']   = age_df['Age'].astype(float) + 1.0   # increment by 1 year
        return age_df
    except Exception:
        logger.warning(f"Could not load Age from players{year-1}")
        return None


def _parse_pos_fallback(pos_str) -> list:
    """Parse 'DH,SP' or 'OF' into ['DH','SP'] or ['OF']."""
    if not isinstance(pos_str, str) or not pos_str.strip():
        return []
    return [p.strip() for p in pos_str.replace('/', ',').split(',') if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Roster builder
# ─────────────────────────────────────────────────────────────────────────────

def build_rosters(
    data: pd.DataFrame,
    owner_col: str = 'Owner',
) -> Dict[str, pd.DataFrame]:
    """
    Split simulation data into per-team roster DataFrames.

    Only includes players with a non-null Owner value.
    """
    owned = data[data[owner_col].notna() & (data[owner_col] != '')].copy()
    rosters: Dict[str, pd.DataFrame] = {}
    for owner, grp in owned.groupby(owner_col):
        rosters[str(owner)] = grp.reset_index(drop=True)

    if not rosters:
        logger.warning(
            "No owned players found. Run the draft first, then re-run the simulation."
        )
    else:
        sizes = {o: len(df) for o, df in rosters.items()}
        logger.info(f"Loaded rosters: {sizes}")

    return rosters


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────────────

def _sim_chunk(args: tuple) -> List[pd.DataFrame]:
    """Worker function for parallel simulation. Must be top-level for pickling."""
    rosters, variance_model, n_sims, n_weeks, optimize_col, seed = args
    sim = SeasonSimulator(
        rosters, variance_model,
        n_sims=n_sims, n_weeks=n_weeks, optimize_col=optimize_col,
    )
    sim.run(seed=seed)
    return sim.sim_records


def run_simulation(
    db_path: str = 'fantasy_data.db',
    year: int = 2026,
    n_sims: int = 1_000,
    n_weeks: int = N_WEEKS,
    n_workers: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    End-to-end parallel simulation runner.

    Splits n_sims across n_workers processes (default 4 = all Pi cores).
    On a Raspberry Pi 5 with 4 cores, 1000 sims runs in ~18 minutes.

    Returns (summary_df, all_sim_records) where all_sim_records is a flat
    list of per-simulation standings DataFrames for deeper analysis.

    Example
    ───────
    summary, records = run_simulation(year=2026, n_sims=1000)
    print(summary[['owner','pts_mean','p_1st','p_top3','p_top6']])

    # Full roto-points distribution for your team
    my_team = 'Lima Time'
    totals = [s.loc[my_team, 'total'] for s in records if my_team in s.index]
    import matplotlib.pyplot as plt
    plt.hist(totals, bins=40)
    plt.title(f'{my_team} roto points distribution ({len(totals)} sims)')
    plt.show()
    """
    from multiprocessing import Pool

    data           = load_simulation_data(db_path, year)
    rosters        = build_rosters(data)
    variance_model = PlayerVarianceModel(data)

    # Split sims across workers
    chunk     = max(1, n_sims // n_workers)
    remainder = n_sims - chunk * n_workers
    chunks    = [chunk + (1 if i < remainder else 0) for i in range(n_workers)]
    rng       = np.random.default_rng(seed)
    seeds     = [int(rng.integers(0, 2**31)) for _ in range(n_workers)]

    args = [
        (rosters, variance_model, c, n_weeks, 'z', s)
        for c, s in zip(chunks, seeds)
    ]

    logger.info(
        f"Running {n_sims} simulations across {n_workers} workers "
        f"({chunk} sims/worker)"
    )

    with Pool(n_workers) as pool:
        chunk_records = pool.map(_sim_chunk, args)

    all_records = [rec for chunk in chunk_records for rec in chunk]

    # Re-summarize from combined records using a temporary simulator
    sim = SeasonSimulator(rosters, variance_model, n_sims=0, n_weeks=n_weeks)
    sim.sim_records = all_records
    summary = sim._summarize()

    return summary, all_records
