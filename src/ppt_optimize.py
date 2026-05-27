"""
Phase 7: PPT 优化后处理
- 首帧延长（pause 效果）
- 年份水印进度条叠加
- 可选裁剪到 16:9 比例
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import YEARS, FPS


def add_year_overlay(
    frame_path: Path,
    year: int,
    total_years: int = len(YEARS),
    font_size: int = 36,
) -> Image.Image:
    """
    在帧上添加年份水印 + 底部进度条。

    返回 PIL Image（RGB）。
    """
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体
    try:
        font_large = ImageFont.truetype("simhei.ttf", font_size)
        font_small = ImageFont.truetype("simhei.ttf", int(font_size * 0.6))
    except Exception:
        try:
            font_large = ImageFont.truetype("msyh.ttf", font_size)
            font_small = ImageFont.truetype("msyh.ttf", int(font_size * 0.6))
        except Exception:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # 年份水印（右上角）
    year_text = str(year)
    bbox = draw.textbbox((0, 0), year_text, font=font_large)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 20
    # 半透明背景
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [w - tw - margin * 2, margin, w - margin, margin + th + 10],
        radius=8, fill=(255, 255, 255, 200),
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text(
        (w - tw - margin - 5, margin + 5), year_text,
        fill=(30, 30, 30), font=font_large,
    )

    # 底部进度条
    bar_h = 6
    bar_y = h - bar_h - 8
    progress = (year - YEARS[0]) / (YEARS[-1] - YEARS[0])
    # 背景条
    draw.rectangle([10, bar_y, w - 10, bar_y + bar_h], fill=(80, 80, 80, 120))
    # 进度条
    fill_w = int((w - 20) * progress)
    draw.rectangle([10, bar_y, 10 + fill_w, bar_y + bar_h], fill=(220, 50, 50))

    # 起止年份标注
    draw.text((12, bar_y - 16), str(YEARS[0]), fill=(180, 180, 180), font=font_small)
    draw.text((w - 40, bar_y - 16), str(YEARS[-1]), fill=(180, 180, 180), font=font_small)

    return img


def extend_first_frame(frame_dir: Path, n_copies: int = 8) -> None:
    """
    复制首帧多次以制造"暂停"效果（在合成动画前调用）。
    """
    first_frame = sorted(frame_dir.glob("frame_*.png"))[0]
    # 临时重命名策略：在序号前插入
    # 简单做法：创建 frame_0000_pauseX.png
    for i in range(n_copies):
        copy_path = frame_dir / f"frame_0000_pause_{i:02d}.png"
        if not copy_path.exists():
            import shutil
            shutil.copy(first_frame, copy_path)


def optimize_all_frames(frame_dir: Path) -> None:
    """
    对帧目录中的所有帧添加年份水印和进度条（原地覆盖）。
    """
    frames = sorted(frame_dir.glob("frame_*.png"))
    for fp in frames:
        # 提取年份
        try:
            year = int(fp.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        img = add_year_overlay(fp, year)
        img.save(fp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPT优化后处理")
    parser.add_argument("frame_dir", type=Path, help="帧目录路径")
    parser.add_argument("--extend", action="store_true", help="首帧延长")
    parser.add_argument("--overlay", action="store_true", help="添加年份水印+进度条")
    args = parser.parse_args()

    if args.extend:
        extend_first_frame(args.frame_dir)
        print(f"首帧延长完成: {args.frame_dir}")
    if args.overlay:
        optimize_all_frames(args.frame_dir)
        print(f"水印添加完成: {args.frame_dir}")
