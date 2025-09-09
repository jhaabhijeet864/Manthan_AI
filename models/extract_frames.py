import argparse
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import cv2

#!/usr/bin/env python3
"""
Extract frames from .mp4 files in a dataset.

Features:
- Recursively scans input directory for .mp4 files
- Extracts every Nth frame or approximates a target FPS
- Optional start/end times, resize, output format, and overwrite control
- Saves frames under output_root/<video_stem>/<video_stem>_000001.jpg

Usage examples:
    python extract_frames.py -i ./datasets -o ./frames
    python extract_frames.py -i ./datasets -o ./frames --stride 5
    python extract_frames.py -i ./datasets -o ./frames --fps 2.0 --start 3 --end 10 --ext png
"""


try:
except ImportError:
        print("OpenCV (cv2) not found. Install with: pip install opencv-python", file=sys.stderr)
        sys.exit(1)


def find_videos(input_path: Path) -> List[Path]:
        if input_path.is_file() and input_path.suffix.lower() == ".mp4":
                return [input_path]
        if not input_path.exists():
                return []
        vids = list(input_path.rglob("*.mp4"))
        # Also handle uppercase extension
        vids += [p for p in input_path.rglob("*.MP4")]
        # Deduplicate and sort
        uniq = sorted({p.resolve() for p in vids})
        return uniq


def ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)


def guess_step_from_fps(src_fps: float, target_fps: float) -> int:
        if src_fps <= 0 or target_fps <= 0:
                return 1
        if target_fps >= src_fps:
                return 1
        step = max(1, int(round(src_fps / target_fps)))
        return step


def resize_frame(frame, width: Optional[int], height: Optional[int]):
        if width is None and height is None:
                return frame
        h, w = frame.shape[:2]
        if width is not None and height is not None:
                return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if width is not None:
                scale = width / float(w)
                return cv2.resize(frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        # height is not None
        scale = height / float(h)
        return cv2.resize(frame, (int(round(w * scale)), height), interpolation=cv2.INTER_AREA)


def extract_frames_from_video(
        video_path: Path,
        out_root: Path,
        stride: int = 1,
        target_fps: float = 0.0,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        out_ext: str = "jpg",
        jpeg_quality: int = 95,
        overwrite: bool = False,
        resize_w: Optional[int] = None,
        resize_h: Optional[int] = None,
        verbose: bool = True,
) -> Tuple[int, int]:
        """
        Returns (saved_frames, total_processed_frames)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
                if verbose:
                        print(f"[ERROR] Cannot open {video_path}")
                return (0, 0)

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Time range -> frame indices
        start_frame = 0
        end_frame = total_frames - 1 if total_frames > 0 else None

        if start_sec is not None and src_fps > 0:
                start_frame = max(0, int(start_sec * src_fps))
        if end_sec is not None and src_fps > 0:
                ef = int(end_sec * src_fps)
                end_frame = ef if end_frame is None else min(end_frame, ef)

        if end_frame is not None and end_frame < start_frame:
                if verbose:
                        print(f"[WARN] end before start for {video_path}, skipping")
                cap.release()
                return (0, 0)

        # Seek to start frame if possible
        if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Determine step
        step = stride if stride and stride > 0 else 1
        if target_fps and target_fps > 0:
                step = guess_step_from_fps(src_fps, target_fps)

        video_stem = video_path.stem.replace(" ", "_")
        out_dir = out_root / video_stem
        ensure_dir(out_dir)

        saved = 0
        processed = 0
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if start_frame > 0 else 0

        # If we started mid-video, set frame_idx accordingly
        if start_frame > 0:
                frame_idx = start_frame

        # Save parameters based on extension
        out_ext = out_ext.lower().strip(".")
        if out_ext not in ("jpg", "jpeg", "png"):
                out_ext = "jpg"
        imwrite_params = []
        if out_ext in ("jpg", "jpeg"):
                imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, jpeg_quality)))]
        elif out_ext == "png":
                # 0 (no compression) to 9
                imwrite_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

        if verbose:
                rng = ""
                if start_sec is not None or end_sec is not None:
                        rng = f" [{start_sec or 0}s - {end_sec if end_sec is not None else 'end'}s]"
                desc = f"{video_path.name}: fps={src_fps:.2f}, frames={total_frames}"
                if step > 1:
                        desc += f", step={step}"
                print(f"[INFO] {desc}{rng}")

        while True:
                if end_frame is not None and frame_idx > end_frame:
                        break
                ret, frame = cap.read()
                if not ret:
                        break

                # processed counts frames we touched in the selected range
                processed += 1

                # Save every 'step' frames relative to absolute index
                if (frame_idx - start_frame) % step == 0:
                        if resize_w is not None or resize_h is not None:
                                frame = resize_frame(frame, resize_w, resize_h)

                        filename = f"{video_stem}_{frame_idx:06d}.{out_ext}"
                        out_path = out_dir / filename
                        if overwrite or (not out_path.exists()):
                                ok = cv2.imwrite(str(out_path), frame, imwrite_params)
                                if not ok and verbose:
                                        print(f"[WARN] Failed to write {out_path}")
                                else:
                                        saved += 1

                frame_idx += 1

        cap.release()
        if verbose:
                print(f"[DONE] {video_path.name} -> {saved} frames saved in {out_dir}")
        return (saved, processed)


def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(description="Extract frames from .mp4 videos in a dataset directory.")
        p.add_argument("-i", "--input", type=str, default="datasets", help="Input file or directory (recursively scans for .mp4)")
        p.add_argument("-o", "--output", type=str, default="frames", help="Output root directory for extracted frames")
        p.add_argument("--stride", type=int, default=1, help="Keep every Nth frame (ignored if --fps > 0)")
        p.add_argument("--fps", type=float, default=0.0, help="Approximate target FPS to sample (overrides --stride if > 0)")
        p.add_argument("--start", type=float, default=None, help="Start time in seconds")
        p.add_argument("--end", type=float, default=None, help="End time in seconds")
        p.add_argument("--ext", type=str, default="jpg", help="Output image extension: jpg or png")
        p.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100) if ext=jpg")
        p.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
        p.add_argument("--width", type=int, default=None, help="Resize width (preserve aspect if height not set)")
        p.add_argument("--height", type=int, default=None, help="Resize height (preserve aspect if width not set)")
        p.add_argument("--quiet", action="store_true", help="Reduce logging")
        return p.parse_args()


def main():
        args = parse_args()
        inp = Path(args.input)
        out_root = Path(args.output)
        ensure_dir(out_root)

        videos = find_videos(inp)
        if not videos:
                print(f"[WARN] No .mp4 files found under {inp}")
                return

        total_saved = 0
        total_processed = 0
        for v in videos:
                saved, processed = extract_frames_from_video(
                        v,
                        out_root=out_root,
                        stride=max(1, args.stride),
                        target_fps=float(args.fps),
                        start_sec=args.start,
                        end_sec=args.end,
                        out_ext=args.ext,
                        jpeg_quality=args.quality,
                        overwrite=args.overwrite,
                        resize_w=args.width,
                        resize_h=args.height,
                        verbose=not args.quiet,
                )
                total_saved += saved
                total_processed += processed

        if not args.quiet:
                print(f"[SUMMARY] Videos: {len(videos)}, Frames processed: {total_processed}, Frames saved: {total_saved}")
                print(f"[OUTPUT] {out_root.resolve()}")


if __name__ == "__main__":
        main()