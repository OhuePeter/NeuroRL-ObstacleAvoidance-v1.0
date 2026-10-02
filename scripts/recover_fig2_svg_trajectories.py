"""
==========================================================
Recover Figure 2 Trajectories from the Hand-Edited SVG

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
paper/figures/fig2_trajectories.svg was hand-finished in
Inkscape, but it still contains the original per-episode
trajectory paths exported from matplotlib (stroke-opacity
0.4, grouped by condition colour) underneath the bold
"mean" overlay lines and text labels. This script:

  1. Parses the raw SVG path data for those individual
     episode traces, grouped by stroke colour -> condition.
  2. Flattens each path's M/L/H/V/C commands into a polyline
     in local SVG units.
  3. Calibrates a 2D affine transform from local SVG units to
     world coordinates using the two obstacle circles and the
     goal circle (whose world positions are known from the
     environment config: obstacles at (4.25,5.0)/(5.75,5.0),
     goal at (5.0,9.0)).
  4. Applies that transform to every recovered episode path
     and writes the result to CSV, one file per episode.

Usage
-----
python scripts/recover_fig2_svg_trajectories.py
==========================================================
"""

import re
from pathlib import Path

import numpy as np

SVG_PATH = Path("paper/figures/fig2_trajectories.svg")
OUTPUT_ROOT = Path("experiments/version_1_0/results/recovered_fig2_svg_trajectories")

# Condition -> stroke colour used for the individual (opacity 0.4) traces,
# identified by cross-referencing the legend swatches and hue progression.
CONDITION_COLOURS = {
    "P0": "#234551",
    "L1": "#784c7f",
    "L2": "#b34c7d",
    "L3": "#cf4759",
    "R1": "#289c8f",
    "R2": "#1e788a",
    "R3": "#0a3a47",
}

# Known world coordinates (environment geometry) used to calibrate the
# SVG -> world affine transform.
OBSTACLE_1_WORLD = (4.25, 5.0)
OBSTACLE_2_WORLD = (5.75, 5.0)
GOAL_WORLD = (5.0, 9.0)


def tokenize_path(d):
    """Tokenize an SVG path 'd' string into (command, [numbers]) pairs."""
    tokens = re.findall(r"([MLHVCZmlhvcz])|(-?\d*\.?\d+(?:e-?\d+)?)", d)
    commands = []
    current_cmd = None
    current_args = []
    for letter, number in tokens:
        if letter:
            if current_cmd is not None:
                commands.append((current_cmd, current_args))
            current_cmd = letter
            current_args = []
        else:
            current_args.append(float(number))
    if current_cmd is not None:
        commands.append((current_cmd, current_args))
    return commands


def flatten_path(d, bezier_samples=10):
    """Flatten M/L/H/V/C (absolute and relative) into a list of (x, y) points."""
    commands = tokenize_path(d)
    points = []
    x = y = 0.0
    start_x = start_y = 0.0

    def cubic_bezier(p0, p1, p2, p3, n):
        ts = np.linspace(0, 1, n)[1:]
        pts = []
        for t in ts:
            mt = 1 - t
            px = (mt**3) * p0[0] + 3 * (mt**2) * t * p1[0] + 3 * mt * (t**2) * p2[0] + (t**3) * p3[0]
            py = (mt**3) * p0[1] + 3 * (mt**2) * t * p1[1] + 3 * mt * (t**2) * p2[1] + (t**3) * p3[1]
            pts.append((px, py))
        return pts

    for cmd, args in commands:
        is_relative = cmd.islower()
        cmd_upper = cmd.upper()

        if cmd_upper == "M":
            dx, dy = args[0], args[1]
            x, y = (x + dx, y + dy) if is_relative else (dx, dy)
            start_x, start_y = x, y
            points.append((x, y))
            # Extra coordinate pairs after the first are implicit lineto.
            for i in range(2, len(args), 2):
                dx, dy = args[i], args[i + 1]
                x, y = (x + dx, y + dy) if is_relative else (dx, dy)
                points.append((x, y))

        elif cmd_upper == "L":
            for i in range(0, len(args), 2):
                dx, dy = args[i], args[i + 1]
                x, y = (x + dx, y + dy) if is_relative else (dx, dy)
                points.append((x, y))

        elif cmd_upper == "H":
            for v in args:
                x = x + v if is_relative else v
                points.append((x, y))

        elif cmd_upper == "V":
            for v in args:
                y = y + v if is_relative else v
                points.append((x, y))

        elif cmd_upper == "C":
            for i in range(0, len(args), 6):
                c1 = (x + args[i], y + args[i + 1]) if is_relative else (args[i], args[i + 1])
                c2 = (x + args[i + 2], y + args[i + 3]) if is_relative else (args[i + 2], args[i + 3])
                end = (x + args[i + 4], y + args[i + 5]) if is_relative else (args[i + 4], args[i + 5])
                points.extend(cubic_bezier((x, y), c1, c2, end, bezier_samples))
                x, y = end

        elif cmd_upper == "Z":
            points.append((start_x, start_y))
            x, y = start_x, start_y

    return np.array(points)


def bbox_center(points):
    return (points[:, 0].min() + points[:, 0].max()) / 2, (points[:, 1].min() + points[:, 1].max()) / 2


def fit_affine(src_points, dst_points):
    """Fit x' = a*x + b*y + e, y' = c*x + d*y + f from >=3 point pairs (least squares)."""
    n = len(src_points)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i, ((x, y), (xp, yp)) in enumerate(zip(src_points, dst_points)):
        A[2 * i] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i] = xp
        b[2 * i + 1] = yp
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, e, c, d, f = params
    return np.array([[a, bb, e], [c, d, f]])


def apply_affine(matrix, points):
    ones = np.ones((len(points), 1))
    homogeneous = np.hstack([points, ones])
    return homogeneous @ matrix.T


def extract_path_by_id(svg_text, path_id):
    match = re.search(rf'id="{re.escape(path_id)}"', svg_text)
    if match is None:
        raise ValueError(f"path id {path_id} not found")
    # Walk backward from the id to find the start of this <path ... d="...">
    start = svg_text.rfind("<path", 0, match.start())
    end = svg_text.find("/>", match.start())
    fragment = svg_text[start:end]
    d_match = re.search(r'd="([^"]*)"', fragment)
    return d_match.group(1)


def extract_condition_paths(svg_text, colour_hex):
    """Find every <path ...> whose style has this stroke colour and opacity 0.4."""
    pattern = re.compile(
        r'<path\s+style="fill:none;stroke:' + re.escape(colour_hex) +
        r';stroke-width:0\.797292[^"]*stroke-opacity:0\.4"\s+d="([^"]*)"',
    )
    return pattern.findall(svg_text)


def extract_condition_failure_paths(svg_text, colour_hex):
    """Find the genuinely distinct dashed failure-trajectory overlays (not
    the hand-duplicated R3 set, which reuses one identical path repeated
    inside jittered <g> wrappers purely for visual texture)."""
    pattern = re.compile(
        r'<path\s+style="fill:none;stroke:' + re.escape(colour_hex) +
        r';stroke-width:0\.974468[^"]*stroke-dasharray[^"]*"\s+d="([^"]*)"',
    )
    found = pattern.findall(svg_text)
    unique = list(dict.fromkeys(found))  # de-duplicate, preserve order
    return unique


def main():

    svg_text = SVG_PATH.read_text(encoding="utf-8")

    obstacle1_d = extract_path_by_id(svg_text, "path194")
    obstacle2_d = extract_path_by_id(svg_text, "path195")
    goal_d = extract_path_by_id(svg_text, "path192")

    obstacle1_centre = bbox_center(flatten_path(obstacle1_d))
    obstacle2_centre = bbox_center(flatten_path(obstacle2_d))
    goal_centre = bbox_center(flatten_path(goal_d))

    print("Calibration reference points (local SVG units):")
    print(f"  Obstacle 1: {obstacle1_centre} -> world {OBSTACLE_1_WORLD}")
    print(f"  Obstacle 2: {obstacle2_centre} -> world {OBSTACLE_2_WORLD}")
    print(f"  Goal      : {goal_centre} -> world {GOAL_WORLD}")

    affine = fit_affine(
        [obstacle1_centre, obstacle2_centre, goal_centre],
        [OBSTACLE_1_WORLD, OBSTACLE_2_WORLD, GOAL_WORLD],
    )

    # Sanity check: re-apply the fit to the calibration points themselves.
    check = apply_affine(affine, np.array([obstacle1_centre, obstacle2_centre, goal_centre]))
    print("\nCalibration residuals (world units, should be ~0):")
    for name, world, recovered in zip(
        ["Obstacle 1", "Obstacle 2", "Goal"],
        [OBSTACLE_1_WORLD, OBSTACLE_2_WORLD, GOAL_WORLD],
        check,
    ):
        err = np.hypot(recovered[0] - world[0], recovered[1] - world[1])
        print(f"  {name}: recovered {tuple(recovered.round(4))}, error {err:.6f}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\nRecovering per-condition episode trajectories:")
    total = 0
    for condition, colour in CONDITION_COLOURS.items():

        path_ds = extract_condition_paths(svg_text, colour)
        condition_dir = OUTPUT_ROOT / condition
        condition_dir.mkdir(parents=True, exist_ok=True)

        min_x_world = []
        max_x_world = []
        left_route_count = 0
        right_route_count = 0
        for episode_idx, d in enumerate(path_ds):
            local_points = flatten_path(d)
            world_points = apply_affine(affine, local_points)
            min_x_world.append(world_points[:, 0].min())
            max_x_world.append(world_points[:, 0].max())

            # Classify the route using only the obstacle-band crossing
            # (y in [4.5, 5.5]), not the whole trajectory, since the
            # approach/exit legs pass back through the corridor centre.
            band = world_points[(world_points[:, 1] >= 4.5) & (world_points[:, 1] <= 5.5)]
            if len(band) > 0:
                band_x = band[:, 0].mean()
                if band_x < 5.0:
                    left_route_count += 1
                else:
                    right_route_count += 1

            out_path = condition_dir / f"episode_{episode_idx:02d}.csv"
            with open(out_path, "w") as f:
                f.write("x,y\n")
                for px, py in world_points:
                    f.write(f"{px:.6f},{py:.6f}\n")

        total += len(path_ds)
        mean_min = np.mean(min_x_world)
        mean_max = np.mean(max_x_world)
        print(
            f"  {condition} ({colour}): {len(path_ds)} episodes recovered, "
            f"x range (mean of episode min/max): [{mean_min:.2f}, {mean_max:.2f}] "
            f"(obstacle band is [4.25, 5.75]) -- "
            f"{left_route_count} left-route, {right_route_count} right-route episodes"
        )

        # Genuinely distinct dashed failure-trajectory overlays (skips the
        # hand-duplicated R3 set, which is one path copy-pasted 6x for
        # visual texture, not 6 real episodes).
        failure_ds = extract_condition_failure_paths(svg_text, colour)
        for failure_idx, d in enumerate(failure_ds):
            local_points = flatten_path(d)
            world_points = apply_affine(affine, local_points)
            out_path = condition_dir / f"failure_{failure_idx:02d}.csv"
            with open(out_path, "w") as f:
                f.write("x,y\n")
                for px, py in world_points:
                    f.write(f"{px:.6f},{py:.6f}\n")
        if failure_ds:
            total += len(failure_ds)
            print(f"    + {len(failure_ds)} additional distinct failure trajectories recovered")

    print(f"\nTotal episodes recovered: {total}")
    print(f"Saved to: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
