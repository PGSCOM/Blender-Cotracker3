# SPDX-License-Identifier: GPL-3.0-or-later
# Direction masking for CoTracker3 output. Pure torch/numpy-agnostic math
# (works on any array that supports boolean item assignment).


def apply_direction_mask(vis, q_times, direction):
    """Mask out predictions the model did not actually produce.

    - FORWARD: frames before each query frame are model initialization, not
      a prediction -> invisible.
    - BACKWARD: same for frames after each query frame (a backward pass is
      run on the flipped video; in flipped space this is the same rule).
    - BOTH: offline single pass tracks both directions, nothing masked.

    vis: [T, N] boolean array (modified in place, also returned).
    q_times: list of N query frame indices.
    """
    if direction == "BOTH":
        return vis
    for i, q_t in enumerate(q_times):
        if direction == "FORWARD":
            vis[:q_t, i] = False
        else:
            vis[q_t + 1 :, i] = False
    return vis


def online_chunks(T, step):
    """Sliding windows for the online model.

    Returns (ind, end, keep_start, keep_len): chunk [ind..end) of the video,
    of which frames [keep_start..keep_start+keep_len) are kept for the result.
    The first window keeps everything it covered; later windows are trusted
    only for their second half, so every frame is covered exactly once.
    """
    chunks = []
    index = 0
    first = True
    while index < T:
        end = min(index + 2 * step, T)
        n = end - index
        if first:
            keep_start, keep_len = 0, n
            first = False
        else:
            keep_start, keep_len = step, max(0, n - step)
            if keep_len == 0:
                break  # rest of this window was already covered
        chunks.append((index, end, keep_start, keep_len))
        index = end if n < 2 * step else index + step
    return chunks


if __name__ == "__main__":
    # Self-check on the masking and window math.
    import numpy as np

    q = [2, 4]
    vis_all = np.ones((6, 2), dtype=bool)

    forward = apply_direction_mask(vis_all.copy(), q, "FORWARD")
    assert not forward[:2, 0].any() and forward[2:, 0].all()
    assert not forward[:4, 1].any() and forward[4:, 1].all()

    backward = apply_direction_mask(vis_all.copy(), q, "BACKWARD")
    assert not backward[3:, 0].any() and backward[:3, 0].all()
    assert not backward[5:, 1].any() and backward[:5, 1].all()

    both = apply_direction_mask(vis_all.copy(), q, "BOTH")
    assert both.all()

    for T in (1, 5, 8, 16, 17, 30, 100, 101):
        covered = []
        for ind, end, ks, kl in online_chunks(T, step=8):
            assert ind < end and 0 <= ks < end - ind
            covered.append((ind + ks, ind + ks + kl))
        assert covered, T
        assert covered[0][0] == 0 and covered[-1][1] == T, (T, covered)
        for a, b in zip(covered, covered[1:]):
            assert a[1] == b[0], (T, covered)  # contiguous, no overlap
    print("direction_mask.py self-check OK")