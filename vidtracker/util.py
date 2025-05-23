from vidtracker.mil import HaarFeature, WeakClassifier, OnlineMILBoost
import numpy as np
import matplotlib.pyplot as plt

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
    feature_value = haar.compute(ii, 0, 0)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for (x1, y1, w, h), wgt in zip(haar.rects, haar.weights):
        color = 'red' if wgt < 0 else 'blue'
        rect = plt.Rectangle((x1, y1), w + 1, h + 1, edgecolor=color, facecolor='none', linewidth=2)
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
