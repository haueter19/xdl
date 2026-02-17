import ast
import warnings
from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, LpStatus, lpSum, value, PULP_CBC_CMD
from typing import List, Dict, Optional

class Optimized_Lineups:
    """
    Optimize a fantasy baseball lineup for a given owner and player data.

    Uses Integer Linear Programming (ILP) to find the exact optimal assignment
    of players to roster slots, maximizing the chosen scoring column.

    Roster slots:
        Hitters (14): C, 1B, 2B, 3B, SS, MI, CI, OF×5, DH×2
        Pitchers  (9): P×9 (any SP or RP)
    """

    # (slot_name, [eligible position tokens])
    HITTER_SLOTS = [
        ('C',   ['C']),
        ('1B',  ['1B']),
        ('2B',  ['2B']),
        ('3B',  ['3B']),
        ('SS',  ['SS']),
        ('MI',  ['2B', 'SS']),
        ('CI',  ['1B', '3B']),
        ('OF1', ['OF']), ('OF2', ['OF']), ('OF3', ['OF']),
        ('OF4', ['OF']), ('OF5', ['OF']),
        ('DH1', []), ('DH2', []),  # any hitter is DH-eligible
    ]

    def __init__(self, owner, data, optimize_col='z',
                 player_col='Player', pos_col='all_pos',
                 owner_col='Owner', type_col='type', maximize=True):
        self.owner: str = owner
        self.player_col: Optional[str] = player_col
        self.pos_col: Optional[str] = pos_col
        self.owner_col: Optional[str] = owner_col
        self.type_col: Optional[str] = type_col
        self.maximize: bool = maximize

        # Normalise optimize_col to a list internally; keep the original for compat
        self.optimize_col: str = optimize_col if isinstance(optimize_col, str) else optimize_col[0]
        self._optimize_cols: List[str] = (
            [optimize_col] if isinstance(optimize_col, str) else list(optimize_col)
        )

        self.data = data.sort_values(self._optimize_cols, ascending=False)

        keep_cols = [player_col, pos_col, type_col] + self._optimize_cols
        self.d = (
            data[data[owner_col] == owner][keep_cols]
            .set_index(player_col)
            .to_dict(orient='index')
        )
        # Normalise all_pos to a list regardless of how it was stored/retrieved
        for v in self.d.values():
            v[pos_col] = self._parse_all_pos(v[pos_col])

        self.p_dict = {k: v for k, v in self.d.items() if 'p' in v[type_col]}
        self.h_dict = {k: v for k, v in self.d.items() if 'h' in v[type_col]}

    # ------------------------------------------------------------------
    # Eligibility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_all_pos(val) -> list:
        """
        Normalise all_pos to a plain Python list regardless of how it arrived.

        Handles three formats seen across the codebase:
          - Already a list:       ['C', '1B', 'DH']
          - Python repr string:   "['C', '1B', 'DH']"
          - Comma-separated str:  "C,1B,DH"
        """
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('['):
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    pass
            return [p.strip() for p in val.split(',') if p.strip()]
        return []

    @staticmethod
    def _is_eligible(all_pos: list, positions: list) -> bool:
        """Return True if the player is eligible for a slot.

        An empty positions list means the slot is unrestricted (e.g. DH).
        """
        if not positions:
            return True
        return any(pos in all_pos for pos in positions)


    # ------------------------------------------------------------------
    # Optimize orchestrator
    # ------------------------------------------------------------------
    
    def optimize(self):
        self._make_pitcher_combos()
        self._make_hitter_combos()
        return

    # ------------------------------------------------------------------
    # ILP solver
    # ------------------------------------------------------------------

    def _solve_ilp(self, player_dict: dict, slots: list, max: bool = True) -> tuple[dict, int] | tuple[None, int]:
        """
        Assign players to slots via lexicographic ILP optimisation.

        When multiple columns are specified in optimize_col, the solver
        maximises the first column, locks that result as a constraint,
        then maximises the second column, and so on.  With a single column
        this reduces to a standard ILP solve.

        Parameters
        ----------
        player_dict : {player_name: {'pos_col': list, col: float, ...}}
        slots       : [(slot_name, [eligible_position_tokens]), ...]

        Returns
        -------
        (assignment, status) where assignment is {slot_name: player_name}
        on success, or (None, status) if the problem is infeasible/unsolved.
        """
        players = list(player_dict.keys())
        slot_names = [s for s, _ in slots]
        n_players = len(players)

        # Constraints carried forward from higher-priority columns:
        #   [(col, achieved_value), ...]
        locked = []

        for col in self._optimize_cols:
            prob = LpProblem("lineup", LpMaximize if max else LpMinimize)

            x = {
                i: {s: LpVariable(f"x_{i}_{s}", cat='Binary')
                    for s in slot_names}
                for i in range(n_players)
            }

            # Objective: maximise *this* column
            prob += lpSum(
                player_dict[players[i]][col] * x[i][s]
                for i in range(n_players)
                for s in slot_names
            )

            # Each slot filled by exactly one player
            for s in slot_names:
                prob += lpSum(x[i][s] for i in range(n_players)) == 1

            # Each player fills at most one slot
            for i in range(n_players):
                prob += lpSum(x[i][s] for s in slot_names) <= 1

            # Eligibility
            for i, player in enumerate(players):
                p_pos = player_dict[player][self.pos_col]
                for s, eligible_pos in slots:
                    if not self._is_eligible(p_pos, eligible_pos):
                        prob += x[i][s] == 0

            # Lock in results from higher-priority columns
            for locked_col, locked_val in locked:
                prob += lpSum(
                    player_dict[players[i]][locked_col] * x[i][s]
                    for i in range(n_players)
                    for s in slot_names
                ) >= locked_val

            prob.solve(PULP_CBC_CMD(msg=0))

            if prob.status != 1:
                return None, prob.status

            # Record the achieved value for this column
            achieved = sum(
                player_dict[players[i]][col] * value(x[i][s])
                for i in range(n_players)
                for s in slot_names
                if value(x[i][s]) is not None and value(x[i][s]) > 0.5
            )
            locked.append((col, achieved))

        # Extract final assignment from the last solve
        assignment = {}
        for s in slot_names:
            for i, player in enumerate(players):
                if value(x[i][s]) is not None and value(x[i][s]) > 0.5:
                    assignment[s] = player
                    break
        return assignment, prob.status

    def _warn_infeasible(self, group: str, player_dict: dict, slots: list, solver_status: int):
        """Warn with a per-slot eligibility breakdown to aid debugging."""
        status_name = LpStatus.get(solver_status, str(solver_status))

        lines = [
            f"Optimized_Lineups: no feasible {group} lineup for '{self.owner}' "
            f"(solver status: {status_name}).",
            "Eligible players per slot:",
        ]
        for s, eligible_pos in slots:
            eligible = [
                p for p in player_dict
                if self._is_eligible(player_dict[p][self.pos_col], eligible_pos)
            ]
            flag = " *** NO ELIGIBLE PLAYERS ***" if not eligible else ""
            lines.append(f"  {s:>4} {eligible_pos}: {eligible}{flag}")

        warnings.warn("\n".join(lines))

    # ------------------------------------------------------------------
    # Public optimisation methods
    # ------------------------------------------------------------------

    def _make_pitcher_combos(self):
        """
        Select the optimal 9-pitcher lineup.

        Because all 9 P slots are interchangeable (any SP/RP fills any slot),
        the greedy top-9 by score is provably optimal — no combinations needed.
        With multiple optimize columns, sorts lexicographically by the tuple.
        """
        sorted_pitchers = sorted(
            self.p_dict,
            key=lambda p: tuple(self.p_dict[p][c] for c in self._optimize_cols),
            reverse=self.maximize,
        )
        lineup = sorted_pitchers[:9]
        self.pitcher_optimized_lineup = {
            f'P{i}': player for i, player in enumerate(lineup, 1)
        }
        self.pitcher_optimized_z = sum(
            self.p_dict[p][self.optimize_col] for p in lineup
        )

    def _make_hitter_combos(self):
        """
        Select the optimal 14-hitter lineup via ILP.

        Finds the exact assignment of players to the 14 hitter slots
        (C, 1B, 2B, 3B, SS, MI, CI, OF×5, DH×2) that maximises the
        total score, respecting all position-eligibility constraints.
        """
        assignment, status = self._solve_ilp(self.h_dict, self.HITTER_SLOTS, self.maximize)
        if assignment is None:
            self._warn_infeasible('hitter', self.h_dict, self.HITTER_SLOTS, status)
            self.hitter_optimized_lineup = {}
            self.hitter_optimized_z = 0
            return

        self.hitter_optimized_lineup = assignment
        self.hitter_optimized_z = sum(
            self.h_dict[p][self.optimize_col]
            for p in assignment.values()
        )
