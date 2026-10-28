"""Train and evaluate classifiers on extracted features.

Examples:
    # main result: ResNet101 + EfficientNetB2, subject-disjoint split
    python scripts/run_experiment.py --config configs/default.yaml

    # backbone comparison
    python scripts/run_experiment.py --backbones vgg16 inceptionv3 \
        --output-dir results/vgg_inception_subject

    # measure the leak: same features, the original frame-level split
    python scripts/run_experiment.py --split frame --output-dir results/resnet_effnet_frame
"""

from __future__ import annotations

import argparse

from gaitcond.experiment import load_config, run_experiment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--features-dir")
    p.add_argument("--output-dir")
    p.add_argument("--backbones", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--split", choices=["subject", "frame"])
    p.add_argument("--pca-components", type=int)
    p.add_argument("--epochs", type=int)
    args = p.parse_args()

    cfg = load_config(args.config, {
        "features_dir": args.features_dir,
        "output_dir": args.output_dir,
        "backbones": args.backbones,
        "models": args.models,
        "split.kind": args.split,
        "pca_components": args.pca_components,
        "train.epochs": args.epochs,
    })
    summary = run_experiment(cfg)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
