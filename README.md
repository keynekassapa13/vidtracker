# VidTracker

[![CI](https://github.com/keynekassapa13/vidtracker/actions/workflows/test.yml/badge.svg)](https://github.com/keyneoei/vidtracker/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/github/license/keynekassapa13/vidtracker)

**VidTracker** is a Python package for object tracking 

---

## Tracker

| Tracker | Paper | Description |
|--------|-------|-------------|
| `DFSTracker` | *Distribution Fields for Tracking* (CVPR 2012) | Smooth histogram fields with spatial/feature domain convolution |
| `MILTracker` | *Visual Tracking with Online Multiple Instance Learning* (CVPR 2009) | Online boosting with Haar features |
| `LKTracker` | Based on Lucas-Kanade optical flow | Tracks feature points and estimates affine transforms |


## Dataset

You can download the original MILTrack dataset (e.g., `cliffbar`) from:  
--> [https://bbabenko.github.io/miltrack.html](https://bbabenko.github.io/miltrack.html)

Extract the dataset into `data/input/` to match the folder structure shown below.

## Project Structure

```lua
.
├── config.json
├── data
│   ├── input
│   │   ├── cliffbar
│   │   │   ├── cliffbar_frames.txt
│   │   │   ├── cliffbar_gt.txt
│   │   │   ├── cliffbar_MIL_TR*.txt
│   │   │   ├── imgs/
│   │   │       └── imgXXXXX.png
│   │   ├── cliffbar.zip
│   └── output/
│   │   ├── cliffbar
│   │   │   └── imgXXXXX.png
├── tests/
└── vidtracker/
    ├── cli.py
    ├── dfs.py
    ├── lk.py
    ├── mil.py
    ├── util.py
    └── video.py

```
---

## Installation

### From PyPI:

```bash
pip install vidtracker
```

### From GitHub:

```bash
pip install git+https://github.com/keyneoei/vidtracker.git
````

### Or locally:

```bash
git clone https://github.com/keyneoei/vidtracker.git
cd vidtracker
pip install .
```

## CLI Usage

```bash
vidtracker --input=data/input/cliffbar/imgs --output=data/output/cliffbar --tracker=DFS --show_frames
```
or

```bash
python -m vidtracker.cli --input=data/input/cliffbar/imgs --output=data/output/cliffbar --tracker=DFS --show_frames
```

## Usage Example (Python)

```python
from vidtracker import DFSTracker, MILTracker, LKTracker

tracker_type = "DFS"    # or "MIL", "LK"
frame = ...             # read video frame
init_bbox = ...         # init bbox
cfg = ...               # configuration (box)

if tracker_type == "DFS":
    tracker = DFSTracker(frame, init_bbox, cfg)
elif tracker_type == "MIL":
    tracker = MILTracker(frame, init_bbox, cfg)
elif tracker_type == "LK":
    tracker = LKTracker(frame, init_bbox, cfg)

# Process subsequent frames
x, y, w, h, angle = tracker.process_frame(next_frame)
```
---

## References

* **DFS**: Sevilla-Lara, L., Learned-Miller, E. (2012). *Distribution Fields for Tracking*. CVPR.
* **MIL**: Babenko, B., Yang, M.-H., & Belongie, S. (2009). *Visual Tracking with Online Multiple Instance Learning*. CVPR.
* **LK**: Lucas, B., & Kanade, T. (1981). *An Iterative Image Registration Technique*. DARPA.


## License

This project is licensed under the [MIT License](LICENSE).
