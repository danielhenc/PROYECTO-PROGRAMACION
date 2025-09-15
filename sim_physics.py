import numpy as np


def resolve_collisions(Par, diametro):
    """Resolve pairwise elastic collision impulse for equal-mass particles,
    corrects overlap and applies simple contagion rule (0<-1).
    Updates Par in place (velocities, positions, colors).
    """
    n = len(Par)
    # snapshot positions/velocities
    pos_prev = np.array([np.array(p.R, dtype=float) for p in Par])
    vel_prev = np.array([np.array(p.V, dtype=float) for p in Par])
    new_vel = vel_prev.copy()

    diam2 = (diametro) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            r12 = pos_prev[i] - pos_prev[j]
            dist2 = np.dot(r12, r12)
            if dist2 <= diam2 and dist2 > 1e-12:
                v12 = vel_prev[i] - vel_prev[j]
                s = np.dot(v12, r12) / (dist2 + 1e-12)
                delta = s * r12
                new_vel[i] -= delta
                new_vel[j] += delta
                # overlap correction applied to current positions
                dist = np.sqrt(dist2)
                overlap = diametro - dist
                if overlap > 0:
                    nvec = r12 / (dist + 1e-12)
                    correction = 0.5 * overlap * nvec
                    Par[i].R = np.array(Par[i].R, dtype=float) + correction
                    Par[j].R = np.array(Par[j].R, dtype=float) - correction
                # simple contagion: if one is infected (1) the other becomes 1
                if getattr(Par[i], 'Color', None) == 0 and getattr(Par[j], 'Color', None) == 1:
                    Par[i].Color = 1
                elif getattr(Par[i], 'Color', None) == 1 and getattr(Par[j], 'Color', None) == 0:
                    Par[j].Color = 1

    # apply new velocities
    for k in range(n):
        Par[k].V = new_vel[k]
