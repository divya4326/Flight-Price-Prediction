import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess
from src.train import train
from src.evaluate import evaluate


def main():
    print("=" * 40)
    print("Step 1: Preprocessing")
    print("=" * 40)
    preprocess()

    print("\n" + "=" * 40)
    print("Step 2: Training")
    print("=" * 40)
    train()

    print("\n" + "=" * 40)
    print("Step 3: Evaluation")
    print("=" * 40)
    evaluate()

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()