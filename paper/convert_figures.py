"""Reconvert all 6 SVGs to PNG with forced white background."""
import cairosvg
import io
import pathlib
import numpy as np
from PIL import Image

SRC = pathlib.Path(r"C:\Users\Peter\Desktop\paper 1 files\figures_newest")
DST = pathlib.Path(r"C:\Users\Peter\Desktop\Adaptive-Neural-Population-Control-v1.0\paper")

MAPPING = [
    ("schematic_newest1.svg",                      "fig1_schematic.png"),
    ("adaptive_route_newest.svg",                  "fig2_trajectories.png"),
    ("velocity_perturbation_effect_new.svg",       "fig3_velocity.png"),
    ("pca_cluster_new.svg",                        "fig4_pca.png"),
    ("goal_direction_newest2.svg",                 "fig5_decoding.png"),
    ("learning_stability_optimisation_newest.svg", "fig6_learning.png"),
]

PAD = 80  # pixels padding around content


def tight_crop(img: Image.Image, pad: int = 80) -> Image.Image:
    """Crop to non-white content bounding box with padding."""
    arr = np.array(img.convert("RGB"))
    # Mask of pixels that are NOT nearly white (threshold 245/255)
    mask = ~np.all(arr > 245, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img  # fully white, nothing to crop
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = arr.shape[:2]
    l = max(0, cmin - pad)
    u = max(0, rmin - pad)
    r = min(w, cmax + pad)
    d = min(h, rmax + pad)
    return img.crop((l, u, r, d))


for svg_name, png_name in MAPPING:
    print(f"Converting {svg_name} ...", flush=True)
    data = cairosvg.svg2png(
        url=str(SRC / svg_name),
        dpi=300,
        background_color="white",
    )
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = tight_crop(img, PAD)
    out = DST / png_name
    img.save(str(out), dpi=(300, 300))
    print(f"  -> {png_name}  {img.size}  {out.stat().st_size // 1024} KB")

print("\nAll done.")
