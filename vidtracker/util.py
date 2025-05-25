from vidtracker.mil import HaarFeature, WeakClassifier, OnlineMILBoost
from vidtracker.dfs import explode, convolve, gaussian_kernel
import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_image(win_w, win_h, is_positive=False):
    """
    Create a random image of given width and height.
    """
    img = np.random.randint(0, 256, size=(win_h, win_w), dtype=np.uint8)
    if is_positive:
        img[win_h//4:3*win_h//4, win_w//4:3*win_w//4] = 200
    return img

def create_integral_image(img):
    """
    Create an integral image from the given image.
    """
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.int32)
    # axis = 0, cumsum along y dir - column wise
    # axis = 1, cumsum along x dir - row wise
    ii[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return ii

def show_haar_feature():
    win_w, win_h = 24, 24
    img = create_image(win_w, win_h)
    ii = create_integral_image(img)

    haar = HaarFeature(win_w, win_h)
    feature_value = haar.compute(ii, 0, 0, win_w=win_w, win_h=win_h)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    for (rx, ry, rw, rh), wgt in zip(haar.rel_rects, haar.weights):
        x = int(rx * win_w)
        y = int(ry * win_h)
        w = min(int(rw * win_w), win_w - x - 1)
        h = min(int(rh * win_h), win_h - y - 1)
        color = 'red' if wgt < 0 else 'blue'
        rect = plt.Rectangle((x, y), w + 1, h + 1, edgecolor=color, facecolor='none', linewidth=2)
        ax.add_patch(rect)

    plt.title(f"Haar Feature Visualization\nFeature Value: {feature_value:.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_weak_classifier():
    win_w, win_h = 24, 24
    num_samples = 50

    # Random Positives Integrals (should be integral images of positive examples)
    pos_integrals = [create_integral_image(create_image(win_w, win_h)) for _ in range(num_samples)]
    # Random Negatives Integrals (should be integral images that do not contain the object)
    neg_integrals = [create_integral_image(create_image(win_w, win_h)) for _ in range(num_samples)]

    haar = HaarFeature(win_w, win_h)
    pos_feats = np.array([haar.compute(ii, 0, 0) for ii in pos_integrals])
    neg_feats = np.array([haar.compute(ii, 0, 0) for ii in neg_integrals])

    feats = np.concatenate([pos_feats, neg_feats])
    labels = np.array([1] * len(pos_feats) + [0] * len(neg_feats))

    clf = WeakClassifier(feature=haar)
    clf.update(feats, labels)

    x_vals = np.linspace(min(feats.min(), -5), max(feats.max(), 5), 200)
    scores = [clf.score(x) for x in x_vals]

    print(f"Positive mean: {clf.mu_pos:.2f}, std: {clf.sigma_pos:.2f}")
    print(f"Negative mean: {clf.mu_neg:.2f}, std: {clf.sigma_neg:.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, scores, label='Log-Odds Score')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Weak Classifier: Log-Odds Score vs. Feature Value")
    plt.xlabel("Feature Value")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_online_MIL_boost():
    win_w, win_h = 24, 24
    milboost = OnlineMILBoost(win_w, win_h, M=30, K=10)
    center = (win_w // 2, win_h // 2)

    # Train on 15 positive and negative frames
    for _ in range(15):
        pos_img = create_image(win_w, win_h, is_positive=True)
        neg_img = create_image(win_w, win_h, is_positive=False)
        ii_pos = create_integral_image(pos_img)
        ii_neg = create_integral_image(neg_img)
        milboost.train_frame(ii=ii_pos, pos_centers=[center], neg_centers=[])
        milboost.train_frame(ii=ii_neg, pos_centers=[], neg_centers=[center])

    test_pos = create_image(win_w, win_h, is_positive=True)
    test_neg = create_image(win_w, win_h, is_positive=False)
    ii_test_pos = create_integral_image(test_pos)
    ii_test_neg = create_integral_image(test_neg)

    score_pos = milboost.score_patch(ii_test_pos, *center)
    score_neg = milboost.score_patch(ii_test_neg, *center)

    print("Score (positive patch):", score_pos)
    print("Score (negative patch):", score_neg)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(test_pos, cmap='gray')
    axs[0].set_title(f"Positive Patch\nScore: {score_pos:.2f}")
    axs[0].axis('off')
    axs[1].imshow(test_neg, cmap='gray')
    axs[1].set_title(f"Negative Patch\nScore: {score_neg:.2f}")
    axs[1].axis('off')
    plt.suptitle("OnlineMILBoost Patch Scoring")
    plt.tight_layout()
    plt.show()

def show_explode_and_convolve():
    patch = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(patch, (50, 50), 20, 180, -1)

    bins = 8
    sigma = 2.0
    df = explode(patch, bins)
    kernel = gaussian_kernel(sigma)

    df_smooth = np.zeros_like(df)
    for k in range(bins):
        df_smooth[..., k] = convolve(df[..., k], kernel)

    plt.figure(figsize=(12, 9))

    plt.subplot(3, bins, 1)
    plt.title("Original")
    plt.imshow(patch, cmap='gray')
    plt.axis('off')

    for k in range(bins):
        plt.subplot(3, bins, bins + k + 1)
        plt.title(f"Bin {k}")
        plt.imshow(df[..., k], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        plt.subplot(3, bins, 2*bins + k + 1)
        plt.title(f"Smoothed {k}")
        plt.imshow(df_smooth[..., k], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_smoothing():
    df = np.zeros((1, 1, 8), dtype=np.float32)
    df[0, 0, 5] = 1.0  
    sigma_f = 1.0
    r = int(3 * sigma_f)
    ax = np.arange(-r, r+1, dtype=np.float32)
    kernel = np.exp(-ax**2 / (2 * sigma_f**2))
    kernel /= np.sum(kernel)
    print(f"ax: {ax}")
    print(f"kernel: {kernel}")

    pad = np.pad(df, ((0, 0), (0, 0), (r, r)), mode='constant')
    print(f"df: {df}")
    print(f"pad: {pad}")

    smoothed = np.zeros(8, dtype=np.float32)
    for i in range(8):
        window = pad[0, 0, i:i + 2*r + 1]
        smoothed[i] = np.dot(window, kernel)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Before Smoothing (One-hot)")
    plt.stem(df[0, 0, :], basefmt=" ")
    plt.xlabel("Bin Index")
    plt.ylim(0, 1.1)

    plt.subplot(1, 2, 2)
    plt.title("After Feature Smoothing")
    plt.stem(smoothed, basefmt=" ")
    plt.xlabel("Bin Index")
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.show()
    