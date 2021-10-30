
from utils import load_cfg, pprint_color, get_parser

def main(yaml_filepath):
    """Example."""
    cfg = load_cfg(yaml_filepath)

    # Print the configuration - just to make sure that you loaded what you
    # wanted to load
    pprint_color(cfg)
    
if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.filename)