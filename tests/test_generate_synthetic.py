import argparse
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from generate_synthetic import build_dataset, range_pair, yolo_line


class SyntheticDatasetTests(unittest.TestCase):
    def test_yolo_line_is_normalized(self):
        self.assertEqual(yolo_line(0, (10, 20, 30, 60), (100, 100)), "0 0.200000 0.400000 0.200000 0.400000")

    def test_range_pair_rejects_reverse_range(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            range_pair("2,1")

    def test_builds_matching_image_and_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backgrounds = root / "backgrounds"
            mannequins = root / "mannequins"
            output = root / "output"
            backgrounds.mkdir()
            mannequins.mkdir()

            Image.new("RGB", (160, 120), "green").save(backgrounds / "field.jpg")
            obj = Image.new("RGBA", (30, 60), (0, 0, 0, 0))
            ImageDraw.Draw(obj).rectangle((5, 5, 25, 55), fill=(255, 0, 0, 255))
            obj.save(mannequins / "mannequin.png")

            args = argparse.Namespace(
                backgrounds=backgrounds,
                mannequins=mannequins,
                output=output,
                count=1,
                start_index=7,
                overwrite=False,
                min_objects=1,
                max_objects=1,
                scale=(0.25, 0.25),
                rotation=(0.0, 0.0),
                brightness=(1.0, 1.0),
                blur=(0.0, 0.0),
                min_visible=1.0,
                class_id=0,
                class_name="mannequin",
                jpeg_quality=90,
                seed=42,
            )
            build_dataset(args)

            image_path = output / "images" / "synthetic_000007.jpg"
            label_path = output / "labels" / "synthetic_000007.txt"
            self.assertTrue(image_path.is_file())
            values = label_path.read_text(encoding="utf-8").split()
            self.assertEqual(values[0], "0")
            self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in values[1:]))

            with self.assertRaises(FileExistsError):
                build_dataset(args)


if __name__ == "__main__":
    unittest.main()
