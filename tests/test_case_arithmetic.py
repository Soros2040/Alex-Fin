"""Deterministic checks for the published teaching examples; no training or data access."""

from pathlib import Path
import runpy
import unittest


class CaseArithmeticTests(unittest.TestCase):
    def test_fee_first_accounting(self):
        turnover = abs(0.6 - 0.5) + abs(0.2 - 0.3)
        gross = 0.6 * 0.01 + 0.2 * -0.02
        self.assertAlmostEqual(100 * (1 - 0.001 * turnover) * (1 + gross), 100.17996)
        self.assertAlmostEqual(100 * (1 - 0.003 * turnover) * (1 + gross), 100.13988)

    def test_dsr_source_matches_hand_calculation(self):
        # Load this dependency-free module without importing the neural-network package.
        symbols = runpy.run_path(str(Path(__file__).parents[1] / "src/agents/reward.py"))
        state = symbols["DSRState"](a_t=0.001, b_t=0.000101, eta=0.01, eps=0.0, cost_lambda=0.0)
        reward, updated = symbols["compute_dsr_reward"](state, 0.003, 0.0)
        self.assertAlmostEqual(reward, 0.248)
        self.assertAlmostEqual(updated.a_t, 0.00102)
        self.assertAlmostEqual(updated.b_t, 0.00010008)
        self.assertAlmostEqual(state.a_t, 0.001)  # Candidate evaluation preserves input moments.
        negative, _ = symbols["compute_dsr_reward"](state, -0.003, 0.0)
        self.assertAlmostEqual(negative, -0.358)

    def test_drawdown_includes_initial_value(self):
        values = [100.0, 110.0, 99.0, 108.0]
        peak, maximum = values[0], 0.0
        for value in values:
            peak = max(peak, value)
            maximum = max(maximum, 1 - value / peak)
        self.assertAlmostEqual(maximum, 0.10)
        self.assertAlmostEqual(values[-1] / values[0] - 1, 0.08)


if __name__ == "__main__":
    unittest.main()
