#!/usr/bin/env python3
"""Build the local fsaverage5/Schaefer100 surface asset used by panel H."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "results" / "dmf_schaefer100" / "schaefer100_fsaverage5_surface.npz"
)
CBIG_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/"
    "FreeSurfer5.3/fsaverage5/label"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    try:
        import nibabel as nib
        from nibabel.freesurfer.io import read_annot
        from nilearn import datasets
    except ImportError as error:
        raise ImportError(
            "Surface preparation requires nibabel and nilearn; install them with "
            "`pip install nibabel nilearn`. The generated NPZ has no such dependency."
        ) from error

    args = parse_args()
    output = args.output.resolve()
    cache_dir = (args.cache_dir or output.parent / "surface_source").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5", data_dir=cache_dir)

    payload: dict[str, np.ndarray] = {
        "atlas": np.asarray("Schaefer2018_100Parcels_7Networks_order"),
        "mesh": np.asarray("fsaverage5"),
        "annotation_source": np.asarray(CBIG_BASE),
    }
    for hemisphere, short in (("left", "lh"), ("right", "rh")):
        annotation = cache_dir / f"{short}.Schaefer2018_100Parcels_7Networks_order.annot"
        if not annotation.exists():
            urlretrieve(f"{CBIG_BASE}/{annotation.name}", annotation)
        labels, _color_table, names = read_annot(annotation, orig_ids=False)
        coordinates, faces = nib.load(
            str(getattr(fsaverage, f"infl_{hemisphere}"))
        ).agg_data()
        sulc = nib.load(str(getattr(fsaverage, f"sulc_{hemisphere}"))).agg_data()
        decoded_names = np.asarray([name.decode("utf-8") for name in names])
        if labels.shape != (10242,) or len(decoded_names) != 51:
            raise ValueError(f"Unexpected {hemisphere} Schaefer/fsaverage5 dimensions.")
        payload.update(
            {
                f"{hemisphere}_coordinates": np.asarray(coordinates, dtype=np.float32),
                f"{hemisphere}_faces": np.asarray(faces, dtype=np.int32),
                f"{hemisphere}_sulc": np.asarray(sulc, dtype=np.float32),
                f"{hemisphere}_vertex_labels": np.asarray(labels, dtype=np.int16),
                f"{hemisphere}_label_names": decoded_names,
            }
        )

    parcel_names = set(payload["left_label_names"][1:]) | set(payload["right_label_names"][1:])
    if len(parcel_names) != 100:
        raise ValueError(f"Expected 100 unique parcel names, got {len(parcel_names)}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(output)


if __name__ == "__main__":
    main()
