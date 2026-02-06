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

    parser.add_argument(
        "--mode",
        required=True,
        choices=["enroll", "recognize", "inspect"]
    )

    parser.add_argument(
        "--dataset",
        help="Path to dataset for batch enrollment"
    )

    parser.add_argument(
        "--user_folder",
        help="Path to a single user's folder for enrollment"
    )

    parser.add_argument(
        "--image",
        help="Image path for recognition"
    )

    args = parser.parse_args()

    engine = FaceEngine()

    # -------------------------------------------------
    # ENROLL
    # -------------------------------------------------

    if args.mode == "enroll":

        if args.user_folder is not None:
            report = engine.enroll_user(args.user_folder)

            print("\n✅ Single User Enrollment Report:\n")
            print(report)

        else:
            if not args.dataset:
                raise ValueError(
                    "Provide either --dataset for batch "
                    "or --user_folder for single enrollment."
                )

            report = engine.enroll_dataset(args.dataset)

            print("\n✅ Batch Enrollment Report:\n")
            print(report)

    # -------------------------------------------------
    # INSPECT DB
    # -------------------------------------------------

    elif args.mode == "inspect":

        records = engine.list_embeddings()

        print(tabulate(records, headers="keys"))

    # -------------------------------------------------
    # RECOGNIZE
    # -------------------------------------------------

    else:

        if not args.image:
            raise ValueError("Provide --image for recognition.")

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
