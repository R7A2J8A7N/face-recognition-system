import argparse
import time
from pathlib import Path

import cv2
from tabulate import tabulate

from src.core.face_engine import FaceEngine
from src.utils.image_loader import load_image
from src.utils.visualization import draw_results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True,
                        choices=["enroll", "recognize"])

    parser.add_argument("--dataset")
    parser.add_argument("--image")

    args = parser.parse_args()

    engine = FaceEngine()

    if args.mode == "enroll":

        total = engine.enroll_dataset(args.dataset)

        print(f"\n✅ Stored {total} embeddings.\n")

    else:

        image = load_image(args.image)

        results = engine.recognize(image)

        print(tabulate(results, headers="keys"))

        # -----------------------------
        # SAVE DEBUG IMAGE
        # -----------------------------

        if results:

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            output_image = draw_results(image, results)

            filename = output_dir / f"match_{int(time.time())}.jpg"

            cv2.imwrite(str(filename), output_image)

            print(f"\n✅ Output saved -> {filename}\n")

        else:
            print("\nNo faces matched — nothing saved.\n")


if __name__ == "__main__":
    main()
