"""Load config.yaml once; import CFG everywhere."""
from pathlib import Path
import yaml

CFG = yaml.safe_load((Path(__file__).parents[2] / "config.yaml").read_text())
