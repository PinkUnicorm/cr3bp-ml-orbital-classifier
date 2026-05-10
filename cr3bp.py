import numpy as np
from scipy.integrate import solve_ivp


def cr3bp_equations(t, state, mu):
    """
    Equations of motion for the Circular Restricted Three-Body Problem.
    This particular problem has 3 assumptions; 2 Large Masses and 1 smaller mass
    


    State:
    x, y, vx, vy

    mu:
    mass ratio, for Earth-Moon approximately 0.01215
    """
    x, y, vx, vy = state

    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)

    ax = (
        2 * vy
        + x
        - (1 - mu) * (x + mu) / r1**3
        - mu * (x - 1 + mu) / r2**3
    )

    ay = (
        -2 * vx
        + y
        - (1 - mu) * y / r1**3
        - mu * y / r2**3
    )

    return [vx, vy, ax, ay]


def jacobi_constant(x, y, vx, vy, mu):
    """
    Computes the Jacobi constant.
    """
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)

    potential = (
        x**2
        + y**2
        + 2 * (1 - mu) / r1
        + 2 * mu / r2
    )

    velocity_squared = vx**2 + vy**2

    return potential - velocity_squared


def classify_orbit(solution, mu, escape_radius=3.0, collision_radius=0.03):
    """
    Labels an orbit as:
    0 = stable/bounded
    1 = escape
    2 = collision
    """
    x = solution.y[0]
    y = solution.y[1]

    r_from_origin = np.sqrt(x**2 + y**2)

    r1 = np.sqrt((x + mu) ** 2 + y**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2)

    if np.any(r_from_origin > escape_radius):
        return 1

    if np.any(r1 < collision_radius) or np.any(r2 < collision_radius):
        return 2

    return 0


def integrate_orbit(initial_state, mu, t_max=20.0, n_points=1000):
    """
    Numerically integrates one orbit.
    """
    t_eval = np.linspace(0, t_max, n_points)

    solution = solve_ivp(
        fun=lambda t, state: cr3bp_equations(t, state, mu),
        t_span=(0, t_max),
        y0=initial_state,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    return solution


def make_features(x, y, vx, vy, mu):
    """
    Creates the ML feature vector.
    """
    c = jacobi_constant(x, y, vx, vy, mu)

    r1 = np.sqrt((x + mu) ** 2 + y**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2)

    return [mu, x, y, vx, vy, c, r1, r2]