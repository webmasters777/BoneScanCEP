import argparse
import csv
import os

import matplotlib.pyplot as plt

from analysis import build_analysis, iter_image_files, load_image_bgr


def save_metrics_csv(rows, csv_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_analysis_image(fig, output_dir, fname):
    stem, _ext = os.path.splitext(fname)
    save_name = f"analysis_{stem}.png"
    save_path = os.path.join(output_dir, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def analyze_single_image(image_path, output_dir, save_fig, show_fig):
    fname = os.path.basename(image_path)
    img_bgr = load_image_bgr(image_path)
    fig, metrics = build_analysis(img_bgr, fname)

    saved_path = ""
    if save_fig:
        saved_path = save_analysis_image(fig, output_dir, fname)

    if show_fig:
        plt.show()

    plt.close(fig)

    return fname, metrics, saved_path


def analyze_folder(folder_path, output_root, limit, save_fig):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_dir = os.path.join(output_root, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    count = 0
    for image_path in iter_image_files(folder_path):
        if limit and count >= limit:
            break
        fname, metrics, saved_path = analyze_single_image(
            image_path, output_dir, save_fig=save_fig, show_fig=False
        )
        rows.append(
            {
                "folder": folder_name,
                "image": fname,
                "image_path": image_path,
                "output_path": saved_path,
                "contrast": f"{metrics['contrast']:.4f}",
                "energy": f"{metrics['energy']:.4f}",
                "homogeneity": f"{metrics['homogeneity']:.4f}",
                "mean": f"{metrics['mean']:.2f}",
                "std": f"{metrics['std']:.2f}",
                "median": f"{metrics['median']:.2f}",
                "otsu": f"{metrics['otsu']:.0f}",
            }
        )
        count += 1

    return rows, output_dir


def main():
    parser = argparse.ArgumentParser(description="Run bone scan analysis locally.")
    parser.add_argument("--image", dest="image_path", help="Path to a single image file.")
    parser.add_argument(
        "--folder",
        dest="folders",
        action="append",
        default=[],
        help="Folder containing images. Use multiple times for multiple folders.",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="",
        help="Output folder (optional).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images per folder (0 for all).",
    )
    parser.add_argument(
        "--no-fig",
        action="store_true",
        help="Skip saving analysis figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure for single image runs.",
    )
    args = parser.parse_args()

    if args.image_path and args.folders:
        raise ValueError("Use --image for single image or --folder for batch, not both.")

    if not args.image_path and not args.folders:
        raise ValueError("Provide --image or at least one --folder path.")

    output_root = args.output_dir.strip()
    if not output_root:
        output_root = os.getcwd() if args.image_path else os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_root, exist_ok=True)

    if args.image_path:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image not found: {args.image_path}")

        fname, metrics, saved_path = analyze_single_image(
            args.image_path,
            output_root,
            save_fig=not args.no_fig,
            show_fig=args.show,
        )

        if saved_path:
            print(f"Saved: {saved_path}")

        print(f"\n===== Results for: {fname} =====")
        print(f"  Contrast    : {metrics['contrast']:.4f}")
        print(f"  Energy      : {metrics['energy']:.4f}")
        print(f"  Homogeneity : {metrics['homogeneity']:.4f}")
        print(f"  Mean        : {metrics['mean']:.2f}")
        print(f"  Std         : {metrics['std']:.2f}")
        print(f"  Median      : {metrics['median']:.2f}")
        print(f"  Otsu thr    : {metrics['otsu']:.0f}")
        return

    all_rows = []
    for folder_path in args.folders:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        rows, folder_out = analyze_folder(
            folder_path, output_root, limit=args.limit, save_fig=not args.no_fig
        )
        if rows:
            folder_csv = os.path.join(folder_out, "analysis_metrics.csv")
            save_metrics_csv(rows, folder_csv)
        all_rows.extend(rows)

    if all_rows:
        summary_csv = os.path.join(output_root, "analysis_metrics_all.csv")
        save_metrics_csv(all_rows, summary_csv)
        print(f"Saved metrics: {summary_csv}")


if __name__ == "__main__":
    main()
