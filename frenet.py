import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial import KDTree


def smooth_3d_array(points, num=None, **kwargs):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    points = np.zeros((num, 3))
    if num is None:
        num = len(x)
    w = np.arange(0, len(x), 1)
    sx = UnivariateSpline(w, x, **kwargs)
    sy = UnivariateSpline(w, y, **kwargs)
    sz = UnivariateSpline(w, z, **kwargs)
    wnew = np.linspace(0, len(x), num)
    points[:, 0] = sx(wnew)
    points[:, 1] = sy(wnew)
    points[:, 2] = sz(wnew)
    return points


def calculate_tnb_frame(curve, epsilon=1e-8):
    curve = np.asarray(curve)

    # Calculate T (tangent)
    T = np.gradient(curve, axis=0)
    T_norms = np.linalg.norm(T, axis=1)
    T = T / T_norms[:, np.newaxis]

    # Identify straight segments
    is_straight = T_norms < epsilon

    # Calculate N (normal) for non-straight parts
    dT = np.gradient(T, axis=0)
    N = dT - np.sum(dT * T, axis=1)[:, np.newaxis] * T
    N_norms = np.linalg.norm(N, axis=1)

    # Handle points where the normal is undefined or in straight segments
    undefined_N = (N_norms < epsilon) | is_straight

    if np.all(undefined_N):
        # print("the entire curve is straight")
        # If the entire curve is straight, choose an arbitrary normal
        N = np.zeros_like(T)
        N[:, 0] = T[:, 1]
        N[:, 1] = -T[:, 0]
        N = N / np.linalg.norm(N, axis=1)[:, np.newaxis]
    elif np.any(undefined_N):
        # print("handling straight parts")
        # Only proceed with interpolation if there are any straight parts
        # Find segments of curved and straight parts
        segment_changes = np.where(np.diff(undefined_N))[0] + 1
        segments = np.split(np.arange(len(curve)), segment_changes)

        for segment in segments:
            if undefined_N[segment[0]]:
                # This is a straight segment
                left_curved = np.where(~undefined_N[: segment[0]])[0]
                right_curved = (
                    np.where(~undefined_N[segment[-1] + 1 :])[0] + segment[-1] + 1
                )

                if len(left_curved) > 0 and len(right_curved) > 0:
                    # Interpolate between left and right curved parts
                    left_N = N[left_curved[-1]]
                    right_N = N[right_curved[0]]
                    t = np.linspace(0, 1, len(segment))
                    N[segment] = (1 - t[:, np.newaxis]) * left_N + t[
                        :, np.newaxis
                    ] * right_N
                elif len(left_curved) > 0:
                    # Use normal from left curved part
                    N[segment] = N[left_curved[-1]]
                elif len(right_curved) > 0:
                    # Use normal from right curved part
                    N[segment] = N[right_curved[0]]
                else:
                    # No curved parts found, use arbitrary normal
                    N[segment] = np.array([T[segment[0]][1], -T[segment[0]][0], 0])

                # Ensure N is perpendicular to T
                N[segment] = (
                    N[segment]
                    - np.sum(N[segment] * T[segment], axis=1)[:, np.newaxis]
                    * T[segment]
                )
                N[segment] = (
                    N[segment] / np.linalg.norm(N[segment], axis=1)[:, np.newaxis]
                )
    else:
        # print("no straight parts")
        pass

    # If there are no straight parts, N is already calculated correctly for all points

    # Calculate B (binormal) ensuring orthogonality
    B = np.cross(T, N)

    # Ensure perfect orthogonality through Gram-Schmidt
    N = N - np.sum(N * T, axis=1)[:, np.newaxis] * T
    N = N / np.linalg.norm(N, axis=1)[:, np.newaxis]

    B = B - np.sum(B * T, axis=1)[:, np.newaxis] * T
    B = B - np.sum(B * N, axis=1)[:, np.newaxis] * N
    B = B / np.linalg.norm(B, axis=1)[:, np.newaxis]

    return T, N, B


def get_closest(pc_a, pc_b):
    """
    For each point in pc_a, find the closest point in pc_b
    Returns the distance and index of the closest point in pc_b for each point in pc_a
    Parameters
    ----------
    pc_a : [Mx3]
    pc_b : [Nx3]
    """
    tree = KDTree(pc_b)
    dist, idx = tree.query(pc_a, workers=-1)

    if np.max(idx) >= pc_b.shape[0]:
        raise ValueError("idx is out of range")

    return dist, idx


def straighten_using_frenet(helix, points):
    """
    Straighten the structure based on the helix (skeleton) using the Frenet frame.

    Args:
    - helix (numpy array): Points forming the helix (skeleton).
    - points (numpy array): Points surrounding the helix.

    Returns:
    - straightened_helix (numpy array): Straightened version of the helix.
    - straightened_points (numpy array): Transformed surrounding points.
    """
    # Compute the Frenet frame for the helix
    T, N, B = calculate_tnb_frame(helix)

    # Parameterize the helix based on cumulative distance (arclength)
    deltas = np.diff(helix, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Map helix to a straight line along Z-axis
    straightened_helix = np.column_stack(
        (
            np.zeros_like(cumulative_distances),
            np.zeros_like(cumulative_distances),
            cumulative_distances,
        )
    )

    distances_to_helix, closest_idxs = get_closest(points, helix)
    vectors = points - helix[closest_idxs]
    r = distances_to_helix
    T_closest = T[closest_idxs]
    N_closest = N[closest_idxs]
    B_closest = B[closest_idxs]
    theta = np.arctan2(
        np.einsum("ij,ij->i", vectors, N_closest),
        np.einsum("ij,ij->i", vectors, B_closest),
    )
    phi = np.arccos(np.einsum("ij,ij->i", vectors, T_closest) / r)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = cumulative_distances[closest_idxs] + r * np.cos(phi)
    straightened_points = np.column_stack((x, y, z))

    return straightened_helix, np.array(straightened_points)


def frenet_transformation(pc, skel, lb):
    skel_smooth = smooth_3d_array(skel, num=skel.shape[0] * 100, s=200000)
    skel_trans, pc_trans = straighten_using_frenet(skel_smooth, pc)
    return pc_trans, skel_trans, lb
