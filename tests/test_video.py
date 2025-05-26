import numpy as np
from vidtracker.dfs import explode, gaussian_kernel, convolve

def test_dummy():
    assert 1 + 1 == 2, "Dummy test failed, 1 + 1 should equal 2"

def test_explode_shape():
    patch = np.array([[0, 127], [255, 64]], dtype=np.uint8)
    bins = 8
    df = explode(patch, bins)
    assert df.shape == (2, 2, bins), f"Expected shape (2, 2, {bins}), got {df.shape}"
    assert np.all(df.sum(axis=2) == 1), "Each pixel should belong to exactly one bin"

def test_gaussian_kernel_sum():
    k = gaussian_kernel(1.0)
    assert np.isclose(k.sum(), 1.0), "Gaussian kernel should sum to 1"

def test_convolve_identity():
    img = np.eye(3, dtype=np.float32)
    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[1, 1] = 1.0 
    out = convolve(img, kernel)
    print(f"Output of convolution:\n{out}")
    assert np.allclose(out, img, atol=1e-5), (
        f"FFT convolution should approximate identity. \n"
        f"Diff:\n{out - img}"
    )