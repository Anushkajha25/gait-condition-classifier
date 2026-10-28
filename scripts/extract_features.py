"""Extract frozen-backbone features for every CASIA-B silhouette frame.

Example:
    python scripts/extract_features.py --config configs/default.yaml \
        --data-root /content/CASIA-B --features-dir /content/drive/MyDrive/gait_features
"""

from __future__ import annotations

import argparse

from gaitcond.data import build_index
from gaitcond.experiment import load_config
from gaitcond.features import extract_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-root")
    p.add_argument("--features-dir")
    p.add_argument("--backbones", nargs="+", help="override extract_backbones")
    p.add_argument("--frame-stride", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--limit", type=int, help="only the first N frames (smoke test)")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (tests only)")
    args = p.parse_args()

    cfg = load_config(args.config, {
        "data_root": args.data_root,
        "features_dir": args.features_dir,
        "extract_backbones": args.backbones,
        "frame_stride": args.frame_stride,
        "image_size": args.image_size,
        "extract_batch_size": args.batch_size,
    })

    index = build_index(cfg["data_root"], frame_stride=cfg.get("frame_stride", 1))
    if args.limit:
        index = index.sample(n=min(args.limit, len(index)), random_state=0).reset_index(drop=True)
    print(f"{len(index):,} frames, {index['subject'].nunique()} subjects, "
          f"{index['sequence_id'].nunique():,} sequences")
    print(index["condition"].value_counts().to_string())

    extract_features(
        index,
        cfg["extract_backbones"],
        cfg["features_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("extract_batch_size", 128),
        weights=None if args.weights.lower() == "none" else args.weights,
    )
    print(f"Features written to {cfg['features_dir']}")


if __name__ == "__main__":
    main()
