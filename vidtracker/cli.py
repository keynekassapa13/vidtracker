import argparse
import json
from box import Box

from vidtracker.video import process_video

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="config.json", 
                        help="Path to the config file")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input video frames e.g. input/cliffbar/imgs")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output directory e.g. output/cliffbar")
    parser.add_argument("--tracker", type=str, default="DFS", required=False,
                        choices=["MIL", "DFS", "LK"],
                        help="Type of tracker to use")
    parser.add_argument("--show_frames", action="store_true",
                        help="Show frames during tracking")
    
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = Box(json.load(f), default_box=True, default_box_attr=None)
    
    cfg.INPUT.PATH = args.input
    cfg.OUTPUT.PATH = args.output
    cfg.TRACKER = args.tracker
    if args.show_frames:
        cfg.SHOW_FRAMES = True

    print(f"Config: {cfg}")
    process_video(cfg)

if __name__ == "__main__":
    main()