import ast
import warnings
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD


class Optimized_Lineups:
    """
    Optimize a fantasy baseball lineup for a given owner and player data.

    Uses Integer Linear Programming (ILP) to find the exact optimal assignment
    of players to roster slots, maximizing the chosen scoring column.

    Roster slots:
        Hitters (14): C, 1B, 2B, 3B, SS, MI, CI, OF×5, DH×2
        Pitchers  (9): P×9 (any SP or RP)

    Position eligibility is read from each player's 'all_pos' field, which is
    stored as a string representation of a list (e.g. "['C', '1B', 'OF']").
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
        ('DH1', ['DH']), ('DH2', ['DH']),
    ]

    def __init__(self, owner, data, optimize_col='z'):
        self.owner = owner
        self.optimize_col = optimize_col
        self.data = data.sort_values(optimize_col, ascending=False)

        self.d = (
            data[data['Owner'] == owner][['Player', 'all_pos', optimize_col, 'type']]
            .set_index('Player')
            .to_dict(orient='index')
        )
        # Normalise all_pos to a list regardless of how it was stored/retrieved
        for v in self.d.values():
            v['all_pos'] = self._parse_all_pos(v['all_pos'])

        self.p_dict = {k: v for k, v in self.d.items() if 'p' in v['type']}
        self.h_dict = {k: v for k, v in self.d.items() if 'h' in v['type']}

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
        """Return True if any of *positions* appears in the player's all_pos list."""
        return any(pos in all_pos for pos in positions)

    # ------------------------------------------------------------------
    # ILP solver
    # ------------------------------------------------------------------

    def _solve_ilp(self, player_dict: dict, slots: list) -> dict | None:
        """
        Assign players to slots maximising the optimize_col score.

        Parameters
        ----------
        player_dict : {player_name: {'all_pos': str, optimize_col: float, ...}}
        slots       : [(slot_name, [eligible_position_tokens]), ...]

        Returns
        -------
        {slot_name: player_name} on success, None if infeasible.
        """
        players = list(player_dict.keys())
        slot_names = [s for s, _ in slots]

        prob = LpProblem("lineup", LpMaximize)

        # x[i][s] = 1  iff  players[i] is assigned to slot s
        x = {
            i: {s: LpVariable(f"x_{i}_{s}", cat='Binary')
                for s in slot_names}
            for i in range(len(players))
        }

        # Objective: maximise total score
        prob += lpSum(
            player_dict[players[i]][self.optimize_col] * x[i][s]
            for i in range(len(players))
            for s in slot_names
        )

        # Each slot filled by exactly one player
        for s in slot_names:
            prob += lpSum(x[i][s] for i in range(len(players))) == 1

        # Each player fills at most one slot
        for i in range(len(players)):
            prob += lpSum(x[i][s] for s in slot_names) <= 1

        # Eligibility: fix ineligible player-slot pairs to 0
        for i, player in enumerate(players):
            p_pos = player_dict[player]['all_pos']
            for s, eligible_pos in slots:
                if not self._is_eligible(p_pos, eligible_pos):
                    prob += x[i][s] == 0

        prob.solve(PULP_CBC_CMD(msg=0))

        if prob.status != 1:   # 1 = Optimal
            return None

        assignment = {}
        for s in slot_names:
            for i, player in enumerate(players):
                if value(x[i][s]) is not None and value(x[i][s]) > 0.5:
                    assignment[s] = player
                    break
        return assignment

    def _warn_infeasible(self, group: str, player_dict: dict, slots: list):
        """Emit a warning listing which slots have no eligible player."""
        empty_slots = [
            s for s, eligible_pos in slots
            if not any(
                self._is_eligible(player_dict[p]['all_pos'], eligible_pos)
                for p in player_dict
            )
        ]
        msg = f"Optimized_Lineups: no feasible {group} lineup for '{self.owner}'."
        if empty_slots:
            msg += f" Slots with no eligible player: {empty_slots}"
        warnings.warn(msg)

    # ------------------------------------------------------------------
    # Public optimisation methods
    # ------------------------------------------------------------------

    def _make_pitcher_combos(self):
        """
        Select the optimal 9-pitcher lineup.

        Because all 9 P slots are interchangeable (any SP/RP fills any slot),
        the greedy top-9 by score is provably optimal — no combinations needed.
        """
        sorted_pitchers = sorted(
            self.p_dict,
            key=lambda p: self.p_dict[p][self.optimize_col],
            reverse=True,
        )
        self.pitcher_optimized_lineup = sorted_pitchers[:9]
        self.pitcher_optimized_z = sum(
            self.p_dict[p][self.optimize_col]
            for p in self.pitcher_optimized_lineup
        )

    def _make_hitter_combos(self):
        """
        Select the optimal 14-hitter lineup via ILP.

        Finds the exact assignment of players to the 14 hitter slots
        (C, 1B, 2B, 3B, SS, MI, CI, OF×5, DH×2) that maximises the
        total score, respecting all position-eligibility constraints.
        """
        assignment = self._solve_ilp(self.h_dict, self.HITTER_SLOTS)
        if assignment is None:
            self._warn_infeasible('hitter', self.h_dict, self.HITTER_SLOTS)
            self.hitter_optimized_lineup = []
            self.hitter_optimized_z = 0
            return

        self.hitter_optimized_lineup = list(assignment.values())
        self.hitter_optimized_z = sum(
            self.h_dict[p][self.optimize_col]
            for p in self.hitter_optimized_lineup
        )
