"""Project entrypoint.

Running this module launches the training pipeline in models/train.py.
"""

from runpy import run_module


if __name__ == "__main__":
	run_module("models.train", run_name="__main__")
