from dataclasses import dataclass


@dataclass
class DSRState:
    a_t: float = 0.0
    b_t: float = 0.0
    eta: float = 2.0 / 253.0
    eps: float = 1e-8
    cost_lambda: float = 1.0


def compute_dsr_reward(state: DSRState, simple_return: float, turnover: float) -> tuple[float, DSRState]:
    r_t = simple_return
    dsr = (
        state.b_t * (r_t - state.a_t)
        - 0.5 * state.a_t * (r_t * r_t - state.b_t)
    ) / ((state.b_t - state.a_t * state.a_t + state.eps) ** 1.5)
    reward = dsr - state.cost_lambda * turnover
    reward = max(-10.0, min(10.0, reward))
    next_state = DSRState(
        a_t=state.a_t + state.eta * (r_t - state.a_t),
        b_t=state.b_t + state.eta * (r_t * r_t - state.b_t),
        eta=state.eta,
        eps=state.eps,
        cost_lambda=state.cost_lambda,
    )
    return reward, next_state
