from vidtracker.mil import HaarFeature, WeakClassifier
import numpy as np
import matplotlib.pyplot as plt

def create_image(win_w, win_h):
    """
    Create a random image of given width and height.
    """
    return np.random.randint(0, 256, size=(win_h, win_w), dtype=np.uint8)

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