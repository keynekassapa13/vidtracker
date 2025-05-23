import argparse
import json
from box import Box

from vidtracker.video import process_video

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="config.json", 
                        help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = Box(json.load(f), default_box=True, default_box_attr=None)
    
    print(f"Config: {cfg}")
    process_video(cfg)

if __name__ == "__main__":
    main()