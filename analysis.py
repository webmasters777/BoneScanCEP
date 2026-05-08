import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_image_files(folder_path):
    for root, _dirs, files in os.walk(folder_path):
        for name in sorted(files):
            _, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTENSIONS:
                yield os.path.join(root, name)


def load_image_bgr(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img_bgr


def build_analysis(img_bgr, fname="image"):
    if img_bgr is None:
        raise ValueError("Input image is empty or unreadable.")

    original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(gray, 5)
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gaussian)
    norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gx = cv2.Sobel(norm, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
    canny = cv2.Canny(norm, 50, 150)

    otsu_val, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, hotspot_mask = cv2.threshold(norm, int(otsu_val * 1.1), 255, cv2.THRESH_BINARY)
    hotspot_display = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    hotspot_display[hotspot_mask == 255] = [220, 0, 0]

    glcm = graycomatrix(norm, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    lbp_map = local_binary_pattern(norm, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp_map.ravel(), bins=26, range=(0, 26), density=True)

    mean_val = float(norm.mean())
    std_val = float(norm.std())
    median_val = float(np.median(norm))

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(f"Analysis | {fname}", fontsize=14, fontweight="bold", y=0.98)

    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.3)

    for ax, im, title, cmap in zip(
        [fig.add_subplot(gs[0, i]) for i in range(4)],
        [original, gray, median, gaussian],
        ["Original", "Gray", "Median", "Gaussian"],
        [None, "gray", "gray", "gray"],
    ):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax, im, title in zip(
        [fig.add_subplot(gs[1, i]) for i in range(4)],
        [enhanced, norm, sobel, canny],
        ["Enhanced Image", "Normalized", "Sobel", "Canny"],
    ):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax_thr = fig.add_subplot(gs[2, 0])
    ax_thr.imshow(thresh, cmap="gray")
    ax_thr.set_title("Threshold", fontsize=10)
    ax_thr.axis("off")

    ax_hot = fig.add_subplot(gs[2, 1])
    ax_hot.imshow(hotspot_display)
    ax_hot.set_title("Hotspot", fontsize=10)
    ax_hot.axis("off")

    ax_h1 = fig.add_subplot(gs[2, 2])
    hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax_h1.bar(range(256), hist_orig.ravel(), color="blue", width=1.0, alpha=0.8)
    ax_h1.set_title("Original Histogram", fontsize=10)
    ax_h1.set_xlim([0, 255])
    ax_h1.set_xlabel("Pixel Value", fontsize=8)
    ax_h1.set_ylabel("Frequency", fontsize=8)
    ax_h1.tick_params(labelsize=7)

    ax_h2 = fig.add_subplot(gs[2, 3])
    hist_enh = cv2.calcHist([norm], [0], None, [256], [0, 256])
    ax_h2.bar(range(256), hist_enh.ravel(), color="green", width=1.0, alpha=0.8)
    ax_h2.set_title("Enhanced Histogram", fontsize=10)
    ax_h2.set_xlim([0, 255])
    ax_h2.set_xlabel("Pixel Value", fontsize=8)
    ax_h2.set_ylabel("Frequency", fontsize=8)
    ax_h2.tick_params(labelsize=7)

    ax_glcm = fig.add_subplot(gs[3, 0])
    ax_glcm.axis("off")
    ax_glcm.set_title("GLCM Features", fontsize=10, pad=4)
    tbl1 = ax_glcm.table(
        cellText=[
            ["Contrast", f"{contrast:.4f}"],
            ["Energy", f"{energy:.4f}"],
            ["Homogeneity", f"{homogeneity:.4f}"],
        ],
        colLabels=["Feature", "Value"],
        loc="center",
        cellLoc="center",
    )
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(9)
    tbl1.scale(1.2, 1.6)
    for (row, _col), cell in tbl1.get_celld().items():
        if row == 0:
            cell.set_facecolor("#378ADD")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#E6F1FB" if row % 2 == 0 else "white")

    ax_stat = fig.add_subplot(gs[3, 1])
    ax_stat.axis("off")
    ax_stat.set_title("Statistics", fontsize=10, pad=4)
    tbl2 = ax_stat.table(
        cellText=[
            ["Mean", f"{mean_val:.2f}"],
            ["Std", f"{std_val:.2f}"],
            ["Median", f"{median_val:.2f}"],
            ["Otsu thr", f"{otsu_val:.0f}"],
        ],
        colLabels=["Statistic", "Value"],
        loc="center",
        cellLoc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1.2, 1.6)
    for (row, _col), cell in tbl2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1D9E75")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EAF3DE" if row % 2 == 0 else "white")

    ax_lbp = fig.add_subplot(gs[3, 2:])
    ax_lbp.bar(range(len(lbp_hist)), lbp_hist, color="steelblue", width=0.85, alpha=0.85)
    ax_lbp.set_title("LBP Histogram - Feature Extraction", fontsize=10)
    ax_lbp.set_xlabel("LBP Pattern Code", fontsize=9)
    ax_lbp.set_ylabel("Normalized Frequency", fontsize=9)
    ax_lbp.tick_params(labelsize=8)

    metrics = {
        "contrast": float(contrast),
        "energy": float(energy),
        "homogeneity": float(homogeneity),
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "otsu": float(otsu_val),
    }

    return fig, metrics
