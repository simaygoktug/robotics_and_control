#Robot önündeki referans noktaya dönerek ilerler.

def pure_pursuit(x, y, path, lookahead):
    for target in path:
        dist = np.hypot(target[0]-x, target[1]-y)
        if dist > lookahead:
            return np.arctan2(target[1]-y, target[0]-x)
    return 0
