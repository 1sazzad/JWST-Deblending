from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

root = Path(__file__).resolve().parents[2]
files = sorted((root / "data/raw/fits").glob("*.fits"))

coverage = [
    fits.getdata(p, extname="WHT", memmap=True)[::32, ::32] > 0
    for p in files
]

common = np.logical_and.reduce(coverage)
points = np.argwhere(common)

if len(points) == 0:
    raise RuntimeError("No shared coverage found")

cy, cx = points[len(points) // 2] * 32
y0 = max(0, int(cy) - 256)
x0 = max(0, int(cx) - 256)

print(f"Selected region: x={x0}, y={y0}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, path in zip(axes, files):
    with fits.open(path, memmap=True) as hdul:
        image = hdul["SCI_BKSUB"].data
        cutout = image[y0:y0+512, x0:x0+512].copy()

    valid = cutout[np.isfinite(cutout)]

    if valid.size:
        low, high = np.percentile(valid, [5, 99.5])
        ax.imshow(cutout, origin="lower", cmap="gray",
                  vmin=low, vmax=high)
    else:
        ax.text(0.5, 0.5, "No valid pixels",
                ha="center", transform=ax.transAxes)

    ax.set_title(path.stem.split("_")[5].upper())

plt.tight_layout()
output = root / "results/ceers_preview.png"
plt.savefig(output, dpi=160)
print(f"Saved: {output}")