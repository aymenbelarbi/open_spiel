# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BYRSA as a pure-Python OpenSpiel game.

BYRSA's Pledge is **simultaneous and sealed**: every player commits face-down,
then all reveal at once.  That is not a detail -- the entire Gold/Claim duality
hangs on it, and it is why this harness exists.  OpenSpiel supports
simultaneous moves natively; ``_apply_actions(actions)`` is the joint commit,
which is the Flip.

THIS WRAPPER REIMPLEMENTS NO RULE.  Every rule resolves through
``byrsa_sim.rules``, which implements A2 v1.2 and is the only place a BYRSA
rule legally exists.  If this wrapper and ``byrsa_sim`` ever disagree, **the
wrapper is the bug** (byrsa-cards/CLAUDE.md §9).

Concurrency shapes, per 01 §3 -- BYRSA has four in a single round:
  * Pledge   -- SIMULTANEOUS, sealed until the Flip   -> modelled natively here
  * Rescue   -- sequential, Sufet-first, public       -> modelled natively here
  * Ballot   -- simultaneous, sealed until reveal     -> delegated (see below)
  * flip-buy -- sequential, Sufet order, public       -> delegated

The two decisions 00 §3 assigns to OpenSpiel -- the sealed simultaneous pledge
and its action space -- are modelled as real OpenSpiel phases.  The Decree
phase's Ballot and flip-buy, the Sufet's choices, Notable windows and failure
burns are resolved by a configured ``byrsa_sim`` delegate policy, named by the
``delegate`` game parameter.  That is a modelling boundary, not a rules one:
the delegate calls the same engine functions the native harness calls.

Action space -- modelled on ``blotto`` (00 §2), which enumerates 66 legal
allocations rather than treating "allocate across fields" as intractable.
Pledges are **value-bucketed fractions of the printed per-player cost**, never
per-card subsets.  The decode lives in ``byrsa_sim.action_space`` so that this
game and the RLCard env decode an action index to exactly the same cards --
without which E-12's cross-harness comparison would be measuring decode
differences rather than rules differences.

Chance -- ``DETERMINISTIC``, and the reason is worth stating because it is not
the obvious choice.

00 §5 sketched ``EXPLICIT_STOCHASTIC``, copied from
``iterated_prisoners_dilemma``, whose only chance event is a two-way
continue/stop.  BYRSA's setup deals one card of each suit to each player from
four shuffled piles (A2 §2.2); those outcomes cannot be enumerated.

``SAMPLED_STOCHASTIC`` was the next candidate and is WRONG here for a concrete
reason: pyspiel requires ``GetRNGState``/``SetRNGState`` for a sampled-stochastic
game, a Python game cannot supply them (the C++ base raises before consulting
Python), and without them ``state.clone()`` throws -- which disables ISMCTS,
MCTS, CFR and essentially every OpenSpiel algorithm that searches.

``DETERMINISTIC`` is the accurate declaration for what this class actually
exposes.  The ``seed`` parameter pins the deal and every draw, so the game tree
of a given ``python_byrsa(seed=K)`` instance contains **no chance nodes at
all**: ``current_player()`` never returns CHANCE and ``chance_outcomes()`` is
never called.  BYRSA's real chance lives in the choice of seed -- i.e. across
instances, which is exactly how every experiment in this workspace uses it, and
what makes E-12's exact cross-harness reproduction possible.
"""

import random

import numpy as np
import pyspiel

from byrsa_sim import action_space, observation as byrsa_obs, rules
from byrsa_sim.agents import all_agents  # noqa: F401  (populates the registry)
from byrsa_sim.agents import registry
from byrsa_sim.config import Config

_DEFAULT_PARAMS = {
    "players": 5,
    "seed": 0,
    # Parenthesis-free ON PURPOSE: OpenSpiel round-trips game parameters
    # through its game-string parser on clone(), and "BeliefBot(0.5)" parses as
    # a NESTED GAME.  byrsa_sim.agents.registry accepts this colon spelling.
    "delegate": "BeliefBot:0.5",
    "rounds": 6,
    "decrees": True,
    "ambitions": True,
    "houses": True,
    "notables": True,
    "envoy": True,
}

_PHASE_PLEDGE = 0
_PHASE_RESCUE = 1

_GAME_TYPE = pyspiel.GameType(
    short_name="python_byrsa",
    long_name="Python BYRSA",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=7,
    min_num_players=3,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS,
)

# A2 §5 scoring is bounded by the 277 total Holding value (A3 §1) plus the
# largest Ambition (+7, A3 §4).
_MAX_UTILITY = 284.0


class ByrsaGame(pyspiel.Game):
    """The game, from which states and observers are made."""

    def __init__(self, params=None):
        params = dict(_DEFAULT_PARAMS, **(params or {}))
        n = int(params["players"])
        self._config = Config(
            players=n,
            rounds=int(params["rounds"]),
            decrees=bool(params["decrees"]),
            ambitions=bool(params["ambitions"]),
            houses=bool(params["houses"]),
            notables=bool(params["notables"]),
            envoy=bool(params["envoy"]),
        )
        self._seed = int(params["seed"])
        self._delegate = str(params["delegate"])
        super().__init__(
            _GAME_TYPE,
            pyspiel.GameInfo(
                num_distinct_actions=action_space.NUM_ACTIONS,
                max_chance_outcomes=0,
                num_players=n,
                min_utility=0.0,
                max_utility=_MAX_UTILITY,
                utility_sum=None,
                # Pledge + Rescue decisions per round, across `rounds` rounds.
                max_game_length=int(params["rounds"]) * (1 + n),
            ),
            params,
        )

    @property
    def config(self):
        return self._config

    @property
    def delegate_spec(self):
        return self._delegate

    def new_initial_state(self, seed=None):
        return ByrsaState(self, self._seed if seed is None else seed)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return ByrsaObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params)


class _ScriptedPledge:
    """A byrsa_sim agent whose pledge (and optionally rescue) is dictated by
    OpenSpiel, and whose every other decision defers to the delegate policy.

    This is the whole trick that keeps the wrapper rule-free: OpenSpiel decides
    the pledge, and ``byrsa_sim.rules`` runs the round exactly as it would for
    any other agent.
    """

    def __init__(self, delegate):
        self._d = delegate
        self.name = f"OpenSpiel/{delegate.name}"
        self.sees_hidden_state = False
        self._cards = None
        self._rescue = None

    # -- OpenSpiel drives these ------------------------------------------
    def script_pledge(self, cards):
        self._cards = tuple(cards)

    def script_rescue(self, cards):
        self._rescue = tuple(cards)

    def reset(self, seat, rng):
        self.seat = seat
        self.rng = rng
        self._d.reset(seat, rng)

    def pledge(self, obs):
        if self._cards is None:
            return self._d.pledge(obs)
        cards, self._cards = self._cards, None
        return tuple(c for c in cards if c in obs.hand)

    def rescue(self, obs):
        if self._rescue is None:
            return self._d.rescue(obs)
        cards, self._rescue = self._rescue, None
        return tuple(c for c in cards if c in obs.hand)

    # -- everything else is the delegate's ---------------------------------
    def __getattr__(self, item):
        # `copy.deepcopy` and pickle probe for __deepcopy__, __reduce_ex__,
        # __getstate__ &c BEFORE __init__ has run, so a naive
        # `getattr(self._d, item)` asks for `_d`, which re-enters __getattr__,
        # which asks for `_d`... -> RecursionError, and clone() dies with it.
        if item.startswith("__") or item == "_d":
            raise AttributeError(item)
        try:
            return getattr(self.__dict__["_d"], item)
        except KeyError:
            raise AttributeError(item) from None


class ByrsaState(pyspiel.State):
    """Wraps a live ``byrsa_sim.state.GameState``."""

    def __init__(self, game, seed):
        super().__init__(game)
        self._game = game
        self._rng = random.Random(seed)
        cfg = game.config
        self._st = rules.setup(cfg, self._rng)
        self._delegates = [registry.make(game.delegate_spec) for _ in range(cfg.players)]
        self._agents = [_ScriptedPledge(d) for d in self._delegates]
        # rules.bind_agents is the CANONICAL seeding recipe, shared by all three
        # harnesses.  Deriving agent RNGs from self._rng would advance the game
        # stream and change later deck shuffles, so the harnesses would diverge
        # for a reason that has nothing to do with the rules.
        rules.bind_agents(self._st, self._agents, seed)
        self._phase = _PHASE_PLEDGE
        self._rescue_order = []
        self._rescue_idx = 0
        self._running_total = 0
        self._returns = None
        self._history_str = []
        rules.begin_round(self._st, self._agents)

    # ---- OpenSpiel API --------------------------------------------------
    def current_player(self):
        if self._st.game_over:
            return pyspiel.PlayerId.TERMINAL
        if self._phase == _PHASE_PLEDGE:
            # THE FLIP: every seat commits at once, sealed until reveal.
            return pyspiel.PlayerId.SIMULTANEOUS
        return self._rescue_order[self._rescue_idx]

    def _legal_actions(self, player):
        assert player >= 0
        phase = "pledge" if self._phase == _PHASE_PLEDGE else "rescue"
        obs = self._obs(player, phase)
        return action_space.legal_actions(obs, phase)

    def _obs(self, player, phase):
        extra = {}
        if phase == "rescue":
            extra["gap"] = max(0, self._st.cost - self._running_total)
        return byrsa_obs.build(self._st, player, phase, **extra)

    def _apply_actions(self, actions):
        """THE JOINT COMMIT -- the Flip (A2 §3 Step 3).

        Every seat's cards are decoded from an observation built BEFORE any
        commit is applied, so no seat can condition on another's pledge.  Then
        byrsa_sim.rules.step_pledge runs the phase exactly as it does natively.
        """
        assert self._phase == _PHASE_PLEDGE and not self._st.game_over
        pub = byrsa_obs.public_snapshot(self._st)
        for p, a in enumerate(actions):
            obs = byrsa_obs.build(self._st, p, "pledge", pub=pub)
            self._agents[p].script_pledge(action_space.decode_pledge(obs, int(a)))
        self._history_str.append(
            "P:" + ",".join(action_space.ACTION_NAMES[int(a)] for a in actions))

        total = rules.step_pledge(self._st, self._agents)
        self._running_total = total
        if total >= self._st.cost:
            self._finish_round()
            return
        # A2 §3b -- exactly one lap, starting with the Sufet, clockwise.
        self._phase = _PHASE_RESCUE
        self._rescue_order = [(self._st.sufet + i) % self._st.n
                              for i in range(self._st.n)]
        self._rescue_idx = 0

    def _apply_action(self, action):
        """One seat's Rescue turn -- a genuinely sequential phase where
        current_player() returns a real seat (00 §5)."""
        assert self._phase == _PHASE_RESCUE and not self._st.game_over
        p = self._rescue_order[self._rescue_idx]
        obs = self._obs(p, "rescue")
        self._agents[p].script_rescue(action_space.decode_rescue(obs, int(action)))
        self._running_total = rules.rescue_one(
            self._st, self._agents, p, self._running_total)
        self._history_str.append(f"R{p}:{action_space.ACTION_NAMES[int(action)]}")
        self._rescue_idx += 1
        if (self._rescue_idx >= len(self._rescue_order)
                or self._running_total >= self._st.cost):
            self._finish_round()

    def _finish_round(self):
        """Steps 3c-5 and the next round, all inside byrsa_sim.rules."""
        st, ag = self._st, self._agents
        contained = self._running_total >= st.cost
        rules.step_resolution(st, ag, contained)
        if st.standing == 0:
            st.destroyed = True
            st.game_over = True
        if not st.game_over:
            rules.step_sufet(st, ag)
        st.forbidden_suit = -1
        st.required_suit = -1
        if not st.game_over:
            rules.step_decree(st, ag)
        if st.round >= st.max_rounds:
            st.game_over = True
        if st.game_over:
            res = rules.score(st)
            self._returns = [float(x) for x in res["totals"]]
        else:
            rules.begin_round(st, ag)
            self._phase = _PHASE_PLEDGE
            self._running_total = 0

    def _action_to_string(self, player, action):
        return action_space.ACTION_NAMES[int(action)]

    def is_terminal(self):
        return bool(self._st.game_over)

    def returns(self):
        """A2 §5 final scores.  Binary and total: Claims score iff the city
        stands, the hand scores iff it falls, Ambitions score in both."""
        if self._returns is None:
            return [0.0] * self._st.n
        return list(self._returns)

    def rewards(self):
        return self.returns() if self.is_terminal() else [0.0] * self._st.n

    # ---- inspection -----------------------------------------------------
    @property
    def byrsa_state(self):
        """The underlying byrsa_sim state -- for E-12 reconciliation only."""
        return self._st

    def information_state_string(self, player=None):
        if player is None:
            player = self.current_player()
        return _info_string(self._st, player, self._history_str)

    def observation_string(self, player=None):
        return self.information_state_string(player)

    def __str__(self):
        st = self._st
        return (f"BYRSA r{st.round}/{st.max_rounds} "
                f"pillars={''.join('1' if x else '0' for x in st.pillars)} "
                f"phase={'PLEDGE' if self._phase == _PHASE_PLEDGE else 'RESCUE'}")


def _info_string(st, player, history):
    """Only what A1 §2 lets this seat see."""
    if player is None or player < 0:
        return " | ".join(history[-4:])
    obs = byrsa_obs.build(st, player, "info")
    return (f"seat={player} r={st.round} "
            f"hand={sorted(obs.hand)} "
            f"backs={[list(x) for x in obs.hand_suit_counts]} "
            f"sizes={list(obs.hand_sizes)} "
            f"claims={[sum(1 for _ in c) for c in obs.claims]} "
            f"pillars={''.join('1' if x else '0' for x in obs.pillars)} "
            f"| " + " ".join(history[-4:]))


class ByrsaObserver:
    """Observer conforming to the PyObserver interface.

    Deliberately string-only.  A1 §2 makes hand *values* hidden and hand
    *composition by suit back* semi-public; a dense tensor that flattened both
    would be the exact information leak 01 §5 exists to prevent.
    """

    def __init__(self, iig_obs_type, params):
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {}

    def set_from(self, state, player):
        pass

    def string_from(self, state, player):
        return state.information_state_string(player)


pyspiel.register_game(_GAME_TYPE, ByrsaGame)
