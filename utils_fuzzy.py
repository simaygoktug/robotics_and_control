## Fuzzy primitives (utils_fuzzy.py)
import math

# Triangular and trapezoidal membership helpers

def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def trap(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)

# Aggregation and centroid defuzzification

def centroid(defs):
    """
    defs: list of (mu, (low, high, typ)) where typ in {"S","Z","tri"}
    Approximates each set with simple shapes for speed; returns crisp value.
    """
    NUM_STEPS = 101
    if not defs:
        return 0.0
    xs = []
    mus = []
    lo = min(d[1][0] for d in defs)
    hi = max(d[1][1] for d in defs)
    if hi <= lo:
        return 0.0
    step = (hi - lo) / (NUM_STEPS - 1)
    for i in range(NUM_STEPS):
        x = lo + i * step
        mu_x = 0.0
        for mu, (a, b, typ) in defs:
            if mu <= 0.0:
                continue
            if typ == "tri":
                # peak at mid
                peak = (a + b) / 2.0
                mu_shape = min(mu, tri(x, a, peak, b))
            elif typ == "Z":
                # descending (left high → right low)
                # map to trapezoid for efficiency
                mid = (a + b) / 2.0
                mu_shape = min(mu, trap(x, a, a, mid, b))
            else:  # "S": ascending
                mid = (a + b) / 2.0
                mu_shape = min(mu, trap(x, a, mid, b, b))
            mu_x = max(mu_x, mu_shape)
        xs.append(x)
        mus.append(mu_x)
    denom = sum(mus)
    if denom <= 1e-9:
        return 0.0
    return sum(x * m for x, m in zip(xs, mus)) / denom