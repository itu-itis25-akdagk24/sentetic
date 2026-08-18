from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Klasor bulunamadi: {directory}")
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"Klasorde desteklenen gorsel yok: {directory}")
    return files


def transform_mannequin(
    source: Image.Image,
    background_size: tuple[int, int],
    rng: random.Random,
    scale_range: tuple[float, float],
    rotation_range: tuple[float, float],
    brightness_range: tuple[float, float],
    blur_range: tuple[float, float],
) -> Image.Image:
    obj = source.convert("RGBA")
    alpha_bbox = obj.getchannel("A").getbbox()
    if alpha_bbox is None:
        raise ValueError("Manken PNG tamamen seffaf")
    obj = obj.crop(alpha_bbox)

    bg_w, bg_h = background_size
    target_height = max(2, round(bg_h * rng.uniform(*scale_range)))
    target_width = max(2, round(obj.width * target_height / obj.height))
    # Asiri yatay nesnelerin belleği veya tuvali kontrolsuz buyutmesini engelle.
    if target_width > bg_w * 2:
        ratio = (bg_w * 2) / target_width
        target_width = max(2, round(target_width * ratio))
        target_height = max(2, round(target_height * ratio))
    obj = obj.resize((target_width, target_height), Image.Resampling.LANCZOS)

    rgb = ImageEnhance.Brightness(obj.convert("RGB")).enhance(
        rng.uniform(*brightness_range)
    )
    rgb.putalpha(obj.getchannel("A"))
    obj = rgb.rotate(
        rng.uniform(*rotation_range),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    blur_radius = rng.uniform(*blur_range)
    if blur_radius > 0:
        # RGB ve alfa birlikte yumusatilir; kenarlar arka plana daha dogal karisir.
        obj = obj.filter(ImageFilter.GaussianBlur(blur_radius))
    return obj


def paste_randomly(
    background: Image.Image,
    obj: Image.Image,
    rng: random.Random,
    min_visible: float,
) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    bg_w, bg_h = background.size
    max_out_x = round(obj.width * (1.0 - min_visible))
    max_out_y = round(obj.height * (1.0 - min_visible))
    x_low, x_high = -max_out_x, bg_w - obj.width + max_out_x
    y_low, y_high = -max_out_y, bg_h - obj.height + max_out_y
    # Nesne tuvalden buyukse istenen gorunurluk orani geometrik olarak
    # mumkun olmayabilir. O eksende gorunen alani maksimize ederek ortala.
    x = rng.randint(x_low, x_high) if x_low <= x_high else (bg_w - obj.width) // 2
    y = rng.randint(y_low, y_high) if y_low <= y_high else (bg_h - obj.height) // 2

    left, top = max(0, x), max(0, y)
    right, bottom = min(bg_w, x + obj.width), min(bg_h, y + obj.height)
    if right <= left or bottom <= top:
        return None

    crop = obj.crop((left - x, top - y, right - x, bottom - y))
    visible_bbox = crop.getchannel("A").getbbox()
    if visible_bbox is None:
        return None
    background.alpha_composite(crop, (left, top))
    ax1, ay1, ax2, ay2 = visible_bbox
    return background, (left + ax1, top + ay1, left + ax2, top + ay2)


def yolo_line(class_id: int, bbox: tuple[int, int, int, int], size: tuple[int, int]) -> str:
    x1, y1, x2, y2 = bbox
    width, height = size
    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"


def build_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    backgrounds = image_files(args.backgrounds)
    mannequins = image_files(args.mannequins)
    images_dir = args.output / "images"
    labels_dir = args.output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stems = [f"synthetic_{index:06d}" for index in range(args.start_index, args.start_index + args.count)]
    existing = [
        path
        for stem in stems
        for path in (images_dir / f"{stem}.jpg", labels_dir / f"{stem}.txt")
        if path.exists()
    ]
    if existing and not args.overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        suffix = " ..." if len(existing) > 3 else ""
        raise FileExistsError(
            f"{len(existing)} cikti dosyasi zaten var: {preview}{suffix}. "
            "Uzerine yazmak icin --overwrite kullanin veya --start-index degerini degistirin."
        )

    for index, stem in zip(range(args.count), stems):
        bg_path = rng.choice(backgrounds)
        with Image.open(bg_path) as opened_bg:
            canvas = opened_bg.convert("RGBA")

        labels: list[str] = []
        object_count = rng.randint(args.min_objects, args.max_objects)
        for _ in range(object_count):
            mannequin_path = rng.choice(mannequins)
            try:
                with Image.open(mannequin_path) as opened_obj:
                    obj = transform_mannequin(
                        opened_obj,
                        canvas.size,
                        rng,
                        args.scale,
                        args.rotation,
                        args.brightness,
                        args.blur,
                    )
            except (OSError, ValueError) as error:
                print(f"Uyari: {mannequin_path.name} atlandi ({error})")
                continue
            result = paste_randomly(canvas, obj, rng, args.min_visible)
            if result is not None:
                canvas, bbox = result
                labels.append(yolo_line(args.class_id, bbox, canvas.size))

        canvas.convert("RGB").save(images_dir / f"{stem}.jpg", quality=args.jpeg_quality)
        (labels_dir / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")

    (args.output / "classes.txt").write_text(f"{args.class_name}\n", encoding="utf-8")
    print(f"Tamamlandi: {args.count} gorsel -> {args.output.resolve()}")


def range_pair(value: str) -> tuple[float, float]:
    try:
        low, high = (float(part) for part in value.split(",", maxsplit=1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Deger 'min,max' biciminde olmali") from error
    if low > high:
        raise argparse.ArgumentTypeError("minimum, maksimumdan buyuk olamaz")
    return low, high


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pillow ile sentetik YOLO veri seti uretir.")
    parser.add_argument("--backgrounds", type=Path, default=Path("data/backgrounds"))
    parser.add_argument("--mannequins", type=Path, default=Path("data/mannequins"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Cikti dosya numaralandirmasinin baslangici (varsayilan: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ayni adli mevcut cikti dosyalarinin uzerine yaz.",
    )
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--scale", type=range_pair, default=(0.08, 0.35), metavar="MIN,MAX")
    parser.add_argument("--rotation", type=range_pair, default=(-25.0, 25.0), metavar="MIN,MAX")
    parser.add_argument("--brightness", type=range_pair, default=(0.65, 1.35), metavar="MIN,MAX")
    parser.add_argument("--blur", type=range_pair, default=(0.0, 1.5), metavar="MIN,MAX")
    parser.add_argument("--min-visible", type=float, default=0.7)
    parser.add_argument("--class-id", type=int, default=0, help="YOLO sinif numarasi (tek sinif icin 0).")
    parser.add_argument("--class-name", default="mannequin", help="classes.txt dosyasina yazilacak sinif adi.")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if (
        args.count < 1
        or args.start_index < 0
        or args.min_objects < 0
        or args.max_objects < args.min_objects
        or args.class_id != 0
    ):
        parser.error("count/start-index/nesne sayisi gecersiz veya tek sinif icin class-id 0 degil")
    if not args.class_name.strip() or "\n" in args.class_name or "\r" in args.class_name:
        parser.error("--class-name bos olamaz veya yeni satir iceremez")
    if not 0 < args.min_visible <= 1:
        parser.error("--min-visible 0 ile 1 arasinda olmali")
    if args.scale[0] <= 0 or args.blur[0] < 0 or args.brightness[0] <= 0:
        parser.error("scale/brightness pozitif, blur negatif olmayan degerler olmali")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality 1 ile 100 arasinda olmali")
    return args


if __name__ == "__main__":
    build_dataset(parse_args())
