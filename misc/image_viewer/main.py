#!/usr/bin/env python3
"""
Large-image viewer for arbitrary JPEG/TIFF files.

Usage:
    python large_image_viewer.py IMAGE_PATH

Controls:
    Mouse wheel       Zoom around cursor
    Left-drag         Pan
    Double-click      Reset to fit
    + / -             Zoom in/out
    0                 Reset to fit
    Esc               Quit

Design:
    - Does NOT build a complete image pyramid at startup.
    - Keeps a small fit-to-window preview in RAM.
    - For zoomed views, requests only the visible source region.
    - Uses an in-memory LRU tile cache.
    - For JPEG, Pillow's draft() is used for the initial reduced-resolution decode.
    - For a pyramid TIFF (multiple resolutions already stored in the file,
      detected via tifffile), zoomed-out tiles are read from the closest
      matching stored level -- for tiled levels, only the overlapping TIFF
      tile segments are read and decoded, not the whole level.
    - For a flat (non-pyramid) TIFF, Pillow decodes the whole file once and
      every tile is a crop of that -- there's no cheaper option without
      pyramid levels in the file.
    - Rendering is done with Tkinter, so there are no GUI dependencies beyond Pillow.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
import os
import queue
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import messagebox

try:
    import numpy as np
    import tifffile
except ImportError:
    np = None
    tifffile = None

Image.MAX_IMAGE_PIXELS = None 
TILE_SIZE = 512
CACHE_TILES = 128
PREVIEW_MAX = 1400
MIN_ZOOM = 0.001
MAX_ZOOM = 32.0
ZOOM_FACTOR = 1.25


def format_bytes(n):
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.2f} {unit}"
        size /= 1024


@dataclass
class View:
    # Source-image coordinate at the center of the canvas.
    cx: float
    cy: float
    zoom: float


class TileCache:
    def __init__(self, max_items=CACHE_TILES):
        self.max_items = max_items
        self._cache = OrderedDict()

    def get(self, key):
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def put(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_items:
            _, old = self._cache.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass

    def clear(self):
        for image in self._cache.values():
            try:
                image.close()
            except Exception:
                pass
        self._cache.clear()


class ReducedSource:
    """
    Backend for decoding tile requests more cheaply than the viewer's
    resident `self.source` path. By default this only covers scale>1
    (zoomed-out) requests -- draft() has nothing finer than native to
    offer, so JPEG never sets `covers_native`. A backend that *can* also
    handle level 0 (a pyramid TIFF whose level-0 page is itself tiled, for
    instance) sets `covers_native = True`, letting the viewer skip loading
    self.source at all, even lazily.
    """

    covers_native = False

    def decode_region(self, level, box):
        raise NotImplementedError

    def build_preview(self, preview_max):
        raise NotImplementedError

    def close(self):
        pass


class JpegDraftBackend(ReducedSource):
    """
    JPEG's draft() gives a cheap reduced-resolution decode of the whole
    file, but a JpegImageFile has a single internal tile covering the
    whole image, so crop()+load() on it always decodes the entire
    (draft-scaled) image regardless of the crop box. level_cache decodes
    each level once and keeps it resident so every tile is a cheap crop of
    an already-decoded image instead of a fresh full decode per tile.
    """

    def __init__(self, filename):
        self.filename = filename
        self.level_cache: dict[int, Image.Image] = {}
        self._level_cache_lock = threading.Lock()
        self._native_size_cache = None

    def _get_level_image(self, level):
        cached = self.level_cache.get(level)
        if cached is not None:
            return cached

        with self._level_cache_lock:
            cached = self.level_cache.get(level)
            if cached is not None:
                return cached

            handle = Image.open(self.filename)
            original_size = handle.size
            scale = 2 ** level

            if scale > 1:
                handle.draft(
                    "RGB",
                    (max(1, original_size[0] // scale), max(1, original_size[1] // scale)),
                )

            if handle.mode not in ("RGB", "RGBA"):
                handle = handle.convert("RGB")

            handle.load()
            self.level_cache[level] = handle
            return handle

    def decode_region(self, level, box):
        level_image = self._get_level_image(level)

        # draft() only honors power-of-two ratios, so the box has to be
        # rescaled to whatever resolution the cached level image actually is.
        native_width, native_height = self._native_size()
        draft_scale_x = native_width / level_image.width
        draft_scale_y = native_height / level_image.height
        draft_box = (
            round(box[0] / draft_scale_x),
            round(box[1] / draft_scale_y),
            round(box[2] / draft_scale_x),
            round(box[3] / draft_scale_y),
        )

        region = level_image.crop(draft_box)

        scale = 2 ** level
        target_w = max(1, math.ceil((box[2] - box[0]) / scale))
        target_h = max(1, math.ceil((box[3] - box[1]) / scale))
        if region.size != (target_w, target_h):
            region = region.resize((target_w, target_h), Image.Resampling.BOX)

        return region

    def _native_size(self):
        if self._native_size_cache is None:
            with Image.open(self.filename) as handle:
                self._native_size_cache = handle.size
        return self._native_size_cache

    def build_preview(self, preview_max):
        preview = Image.open(self.filename)
        if preview.mode not in ("RGB", "RGBA"):
            preview = preview.convert("RGB")
        preview.thumbnail((preview_max, preview_max), Image.Resampling.LANCZOS)
        return preview


class PyramidTiffBackend(ReducedSource):
    """
    Backend for TIFFs with real pyramid levels already stored in the file
    (detected via tifffile's `series.is_pyramidal`). Maps the viewer's
    power-of-two virtual levels to the closest available file level and
    reads directly from that level -- for tiled levels, only the TIFF tile
    segments a request actually overlaps get read and decoded, so a coarse
    view never touches most of the level's bytes.

    This includes level 0 (native resolution) whenever that page is itself
    tiled: `covers_native` is set True in that case, so the viewer never
    needs its resident `self.source` fallback at all, at any zoom -- not
    even lazily. If level 0 isn't tiled (unusual, but the format allows
    it), `covers_native` stays False and level 0 requests fall back to the
    resident-source path exactly as they did before pyramid support
    existed.
    """

    def __init__(self, tiff_file, source_width, source_height):
        self._tf = tiff_file  # kept open for the viewer's lifetime
        self.source_width = source_width
        self.source_height = source_height
        # Serializes filehandle seek+read across decode threads -- decode()
        # itself (the CPU-bound part) runs outside this lock.
        self._file_lock = threading.Lock()
        self._whole_level_cache: dict[int, "np.ndarray"] = {}
        self._whole_level_cache_lock = threading.Lock()
        self._levels = self._build_level_table()
        self.covers_native = self._compute_covers_native()

    def _build_level_table(self):
        # One entry per file level, including level 0, so _pick_level
        # naturally also resolves scale == 1 requests to it. Sorted
        # finest-to-coarsest so _pick_level can do a simple linear scan.
        series = self._tf.series[0]
        levels = []
        for level_series in series.levels:
            page = level_series.pages[0]
            level_h, level_w = level_series.shape[0], level_series.shape[1]
            levels.append({
                "page_index": page.index,
                "width": level_w,
                "height": level_h,
                "scale_x": self.source_width / level_w,
                "scale_y": self.source_height / level_h,
            })
        levels.sort(key=lambda lv: lv["scale_x"])
        return levels

    def _compute_covers_native(self):
        level0 = next(
            (lv for lv in self._levels if lv["width"] == self.source_width and lv["height"] == self.source_height),
            None,
        )
        if level0 is None:
            return False
        return self._tf.pages[level0["page_index"]].is_tiled

    def level_count(self):
        # Excludes level 0 in the reported count -- this describes the
        # "extra" reduced-resolution levels beyond native, matching what
        # print_memory_stats's own level 0 row already covers.
        return len(self._levels) - 1

    def _pick_level(self, requested_scale):
        # Finest available file level that's still at least as detailed as
        # requested (scale <= requested_scale); falls back to the coarsest
        # level if even that isn't detailed enough.
        chosen = None
        for lv in self._levels:
            if lv["scale_x"] <= requested_scale:
                chosen = lv
            else:
                break
        return chosen or (self._levels[-1] if self._levels else None)

    def _read_segment(self, page, index, offsets, counts):
        with self._file_lock:
            self._tf.filehandle.seek(offsets[index])
            data = self._tf.filehandle.read(counts[index])
        seg, _pos, _shape = page.decode(data, index, jpegtables=page.jpegtables)
        return None if seg is None else seg[0]

    def _decode_tiled_region(self, page, region_box):
        # Stitches together only the TIFF-internal tile segments
        # overlapping region_box, rather than decoding the whole page --
        # our virtual tile grid doesn't necessarily line up with the
        # file's own tile grid, so a request can straddle a few segments.
        tile_w, tile_h = page.tilewidth, page.tilelength
        cols = page.chunked[1]
        offsets = page.tags["TileOffsets"].value
        counts = page.tags["TileByteCounts"].value
        samples = page.samplesperpixel

        left = max(0, region_box[0])
        top = max(0, region_box[1])
        right = min(page.imagewidth, region_box[2])
        bottom = min(page.imagelength, region_box[3])
        if left >= right or top >= bottom:
            return None

        out = np.zeros((bottom - top, right - left, samples), dtype=np.uint8)

        tc0, tc1 = left // tile_w, (right - 1) // tile_w
        tr0, tr1 = top // tile_h, (bottom - 1) // tile_h

        for tr in range(tr0, tr1 + 1):
            for tc in range(tc0, tc1 + 1):
                index = tr * cols + tc
                if index >= len(offsets):
                    continue
                seg = self._read_segment(page, index, offsets, counts)
                if seg is None:
                    continue

                tile_top, tile_left = tr * tile_h, tc * tile_w
                src_top = max(0, top - tile_top)
                src_left = max(0, left - tile_left)
                src_bottom = min(tile_h, bottom - tile_top)
                src_right = min(tile_w, right - tile_left)
                if src_top >= src_bottom or src_left >= src_right:
                    continue

                dst_top = tile_top + src_top - top
                dst_left = tile_left + src_left - left
                out[dst_top:dst_top + (src_bottom - src_top), dst_left:dst_left + (src_right - src_left)] = (
                    seg[src_top:src_bottom, src_left:src_right]
                )

        return out

    def _get_whole_level(self, level_index, page):
        # Fallback for levels with no internal tiling to exploit: decode
        # the whole level once and cache it -- same cost model as
        # JpegDraftBackend's level_cache, just fed from the file's own
        # precomputed downsample instead of a draft() resample of level 0.
        cached = self._whole_level_cache.get(level_index)
        if cached is not None:
            return cached
        with self._whole_level_cache_lock:
            cached = self._whole_level_cache.get(level_index)
            if cached is not None:
                return cached
            with self._file_lock:
                arr = self._tf.series[0].levels[level_index].asarray()
            self._whole_level_cache[level_index] = arr
            return arr

    def _decode_level_box(self, file_level, region_box):
        page = self._tf.pages[file_level["page_index"]]
        req_left, req_top, req_right, req_bottom = region_box
        clipped = (
            max(0, req_left), max(0, req_top),
            min(page.imagewidth, req_right), min(page.imagelength, req_bottom),
        )

        if page.is_tiled:
            arr = self._decode_tiled_region(page, clipped)
        else:
            level_index = self._levels.index(file_level)
            whole = self._get_whole_level(level_index, page)
            left, top, right, bottom = clipped
            arr = whole[top:bottom, left:right] if left < right and top < bottom else None

        full_w, full_h = req_right - req_left, req_bottom - req_top
        samples = page.samplesperpixel

        # A request that runs past the page's edge (shouldn't happen from
        # the render loop, which already clips to source_width/height, but
        # cheap to get right) should zero-pad rather than stretch what
        # little valid content exists to fill the requested size -- the
        # same as what Pillow's crop() does for the resident-source path.
        if arr is None or arr.shape[:2] != (full_h, full_w):
            canvas = np.zeros((max(full_h, 1), max(full_w, 1), samples), dtype=np.uint8)
            if arr is not None and arr.size:
                off_y = clipped[1] - req_top
                off_x = clipped[0] - req_left
                canvas[off_y:off_y + arr.shape[0], off_x:off_x + arr.shape[1]] = arr
            arr = canvas

        region = Image.fromarray(np.ascontiguousarray(arr))
        if region.mode not in ("RGB", "RGBA"):
            region = region.convert("RGB")
        return region

    def decode_region(self, level, box):
        scale = 2 ** level
        file_level = self._pick_level(scale)
        if file_level is None:
            raise ValueError("no pyramid levels available")

        region_box = (
            round(box[0] / file_level["scale_x"]), round(box[1] / file_level["scale_y"]),
            round(box[2] / file_level["scale_x"]), round(box[3] / file_level["scale_y"]),
        )
        region = self._decode_level_box(file_level, region_box)

        target_w = max(1, math.ceil((box[2] - box[0]) / scale))
        target_h = max(1, math.ceil((box[3] - box[1]) / scale))
        if region.size != (target_w, target_h):
            region = region.resize((target_w, target_h), Image.Resampling.BOX)
        return region

    def build_preview(self, preview_max):
        coarsest = self._levels[-1] if self._levels else None
        if coarsest is None:
            raise ValueError("no pyramid levels available")
        preview = self._decode_level_box(coarsest, (0, 0, coarsest["width"], coarsest["height"]))
        preview.thumbnail((preview_max, preview_max), Image.Resampling.LANCZOS)
        return preview

    def close(self):
        self._tf.close()


def detect_reduced_source(filename, source_format, source_width, source_height):
    """
    Choose the scale>1 decode backend for this file, or None if there
    isn't a cheaper-than-native option available (a flat, non-pyramid
    TIFF, or any format tifffile/numpy aren't installed for -- either way
    the resident-source path still works for it, just without the
    zoomed-out shortcut). Prints its decision either way, since a silent
    fallback to the resident path is otherwise indistinguishable from a
    detection bug -- the only visible symptom of either is memory usage
    that looks like there's no pyramid backend at all.
    """
    if source_format in ("JPEG", "MPO"):
        print("Reduced-resolution source: JPEG draft() decode")
        return JpegDraftBackend(filename)

    if source_format != "TIFF":
        print(f"Reduced-resolution source: none ({source_format} has no cheap zoomed-out path)")
        return None

    if tifffile is None:
        print("Reduced-resolution source: none (tifffile/numpy not installed -- "
              "pyramid TIFF detection disabled, falling back to resident decode)")
        return None

    tf = None
    try:
        tf = tifffile.TiffFile(filename)
        series = tf.series[0]
        if series.is_pyramidal:
            backend = PyramidTiffBackend(tf, source_width, source_height)
            print(f"Reduced-resolution source: pyramid TIFF, {backend.level_count()} stored levels")
            return backend
        print(f"Reduced-resolution source: none (TIFF has {len(series.levels)} level(s) -- not a pyramid)")
        tf.close()
    except Exception as exc:
        # Any failure here just means treating the file as a flat TIFF --
        # a corrupt/unusual pyramid tag is not a reason to refuse to open
        # the file via the resident-source fallback.
        print(f"Reduced-resolution source: none (pyramid detection failed: {exc!r})")
        if tf is not None:
            tf.close()

    return None


class LargeImageViewer:
    def __init__(self, root, filename):
        self.root = root
        self.filename = os.path.abspath(filename)

        self.root.title(f"Large Image Viewer — {os.path.basename(self.filename)}")
        self.root.geometry("1200x800")

        self.canvas = tk.Canvas(root, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.status = tk.Label(root, anchor="w", padx=8)
        self.status.pack(fill="x")

        self.source = None
        self.preview = None
        self.preview_tk = None
        self.tile_cache = TileCache()

        # True only for formats Pillow can partially decode via draft()
        # (JPEG). draft() only ever helps at scale > 1 though — at native
        # zoom (scale == 1) it's a no-op for every format, so that case
        # always uses the resident-source path below too, just loaded
        # lazily (only if the user actually zooms in that far) rather than
        # up front. source_lock guards both the one-time load and
        # concurrent reads of self.source from the decode threads.
        self.reduced_source = None
        self.source_lock = threading.Lock()
        self._source_loaded = False

        self.view = None
        self.fit_zoom = None

        self.dragging = False
        self.last_mouse = None

        self.render_generation = 0
        self._tk_tiles = []
        self._last_canvas_size = (0, 0)

        # Tiles are decoded on background threads. pending_tiles dedupes
        # in-flight/queued requests; tile_queue is how those threads hand
        # finished tiles back, since Tkinter calls must stay on the main
        # thread. Requests beyond what the pool can run right now sit in
        # request_heap, ordered by distance from the viewport center, so the
        # tiles the user is actually looking at get decoded before ones near
        # the prefetch border.
        # The pool is capped small: each decode can briefly need a full-frame
        # amount of memory, so an uncapped or large pool risks running many
        # of those at once and exhausting RAM.
        self.pending_tiles = set()
        self.tile_queue = queue.Queue()
        self.request_heap = []
        self._request_counter = itertools.count()
        self.active_decodes = 0
        self.max_workers = max(2, (os.cpu_count() or 2) - 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="tile-decode")

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("<KeyPress-plus>", lambda e: self.zoom_at_canvas(1.25, self.canvas.winfo_width()/2, self.canvas.winfo_height()/2))
        self.root.bind("<KeyPress-equal>", lambda e: self.zoom_at_canvas(1.25, self.canvas.winfo_width()/2, self.canvas.winfo_height()/2))
        self.root.bind("<KeyPress-minus>", lambda e: self.zoom_at_canvas(1/1.25, self.canvas.winfo_width()/2, self.canvas.winfo_height()/2))
        self.root.bind("<KeyPress-0>", lambda e: self.reset_view())
        self.root.bind("<Double-Button-1>", lambda e: self.reset_view())

        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_at_canvas(ZOOM_FACTOR, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self.zoom_at_canvas(1 / ZOOM_FACTOR, e.x, e.y))

        self.canvas.bind("<ButtonPress-1>", self.begin_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)

        self.root.after(20, self.load_image)
        self.root.after(15, self._drain_tile_queue)

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.reduced_source is not None:
            self.reduced_source.close()
        self.root.destroy()

    def load_image(self):
        try:
            self.source = Image.open(self.filename)

            # Force a stable RGB/RGBA mode.
            if self.source.mode not in ("RGB", "RGBA"):
                self.source = self.source.convert("RGB")

            self.source_width, self.source_height = self.source.size

            self.reduced_source = detect_reduced_source(
                self.filename, self.source.format, self.source_width, self.source_height,
            )

            self.print_memory_stats()

            if self.reduced_source is None:
                # No cheap reduced-resolution decode path exists for this
                # file at all (e.g. a flat, non-pyramid TIFF) — Pillow has
                # to materialize the whole raster to satisfy *any* crop,
                # even a small preview. Pay that cost exactly once, here,
                # rather than having every tile request separately pay it.
                self.source.load()
                self._source_loaded = True

            self.build_preview()
            self.reset_view()

            self.status.config(
                text=f"{self.source_width:,} × {self.source_height:,}   "
                     f"{self.source.mode}   "
                     f"{os.path.basename(self.filename)}"
            )
        except Exception as exc:
            messagebox.showerror("Unable to open image", str(exc))
            self.close()

    def build_preview(self):
        """
        Build only the small image required for the initial fit view, via
        whichever backend can produce one most cheaply. Falls back to
        resizing self.source directly when there's no reduced_source
        backend (self.source is already fully resident by that point).
        """
        if self.reduced_source is not None:
            self.preview = self.reduced_source.build_preview(PREVIEW_MAX)
        else:
            ratio = min(PREVIEW_MAX / self.source_width, PREVIEW_MAX / self.source_height, 1.0)
            preview_w = max(1, round(self.source_width * ratio))
            preview_h = max(1, round(self.source_height * ratio))
            self.preview = self.source.resize((preview_w, preview_h), Image.Resampling.LANCZOS)

    def print_memory_stats(self):
        """
        Estimate the memory cost of fully materializing this image, both as
        a single uncompressed buffer and as a complete tile pyramid (every
        tile at every level, not just what's currently cached). Printed to
        console as a reference point for further performance work.
        """
        bytes_per_pixel = 4 if self.source.mode == "RGBA" else 3

        uncompressed_bytes = self.source_width * self.source_height * bytes_per_pixel

        print()
        print(f"Memory stats for {os.path.basename(self.filename)}")
        print(f"  Dimensions: {self.source_width:,} x {self.source_height:,} ({self.source.mode}, "
              f"{bytes_per_pixel} bytes/px)")
        if isinstance(self.reduced_source, PyramidTiffBackend):
            if self.reduced_source.covers_native:
                residency_note = (
                    f"  (pyramid TIFF detected, {self.reduced_source.level_count()} stored levels — "
                    "level 0 is itself tiled, so even native zoom reads only the requested tile segments; "
                    "self.source is never loaded)"
                )
            else:
                residency_note = (
                    f"  (pyramid TIFF detected, {self.reduced_source.level_count()} stored levels — "
                    "zoomed-out tiles read from those directly, but level 0 isn't tiled so native zoom "
                    "still loads self.source lazily)"
                )
        elif self.reduced_source is not None:
            residency_note = "  (loaded once you zoom to native resolution — draft() covers everything zoomed out)"
        else:
            residency_note = "  (kept resident immediately — no partial-decode path for this file)"
        print(f"  Uncompressed image: {format_bytes(uncompressed_bytes)}{residency_note}")
        print()
        print(f"  {'Level':>5}  {'Scale':>7}  {'Tiles (x * y = n)':>20}  {'Layer memory':>14}")

        pyramid_bytes = 0
        level = 0
        while True:
            scale = 2 ** level
            tile_source_size = TILE_SIZE * scale

            tiles_x = math.ceil(self.source_width / tile_source_size)
            tiles_y = math.ceil(self.source_height / tile_source_size)
            tile_count = tiles_x * tiles_y

            # Tiles clip at the image edges, but together they cover exactly
            # the full image downsampled by `scale` — so the level's total
            # pixel count (and thus memory) can be computed directly rather
            # than by summing every individual tile's clipped size.
            level_w = math.ceil(self.source_width / scale)
            level_h = math.ceil(self.source_height / scale)
            level_bytes = level_w * level_h * bytes_per_pixel
            pyramid_bytes += level_bytes

            tiles_label = f"{tiles_x} * {tiles_y} = {tile_count}"
            print(f"  {level:>5}  1/{scale:<6}  {tiles_label:>20}  {format_bytes(level_bytes):>14}")

            if tiles_x <= 1 and tiles_y <= 1:
                break

            level += 1
            if level > 40:
                break  # safety net; should never trigger for real images

        print()
        print(f"  Full tile pyramid, all levels:         {format_bytes(pyramid_bytes)}")
        print(f"  Total with raw image kept resident:    {format_bytes(pyramid_bytes + uncompressed_bytes)}")
        print(f"  Total without raw image kept resident: {format_bytes(pyramid_bytes)}")
        print()

    def reset_view(self):
        if self.source is None:
            return

        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())

        # Leave a little margin.
        self.fit_zoom = min(
            w / self.source_width,
            h / self.source_height
        ) * 0.98

        self.fit_zoom = max(self.fit_zoom, MIN_ZOOM)

        self.view = View(
            cx=self.source_width / 2,
            cy=self.source_height / 2,
            zoom=self.fit_zoom,
        )
        self.render()

    def source_from_canvas(self, x, y):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        sx = self.view.cx + (x - w / 2) / self.view.zoom
        sy = self.view.cy + (y - h / 2) / self.view.zoom
        return sx, sy

    def canvas_from_source(self, sx, sy):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        x = w / 2 + (sx - self.view.cx) * self.view.zoom
        y = h / 2 + (sy - self.view.cy) * self.view.zoom
        return x, y

    def zoom_at_canvas(self, factor, x, y):
        if self.view is None:
            return

        old_sx, old_sy = self.source_from_canvas(x, y)

        new_zoom = max(
            MIN_ZOOM,
            min(MAX_ZOOM, self.view.zoom * factor)
        )

        self.view.zoom = new_zoom

        # Keep the source point under the cursor stationary.
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        self.view.cx = old_sx - (x - w / 2) / new_zoom
        self.view.cy = old_sy - (y - h / 2) / new_zoom

        self.clamp_center()
        self.render()

    def on_wheel(self, event):
        factor = ZOOM_FACTOR if event.delta > 0 else 1 / ZOOM_FACTOR
        self.zoom_at_canvas(factor, event.x, event.y)

    def begin_pan(self, event):
        self.dragging = True
        self.last_mouse = (event.x, event.y)

    def pan(self, event):
        if not self.dragging or self.view is None:
            return

        dx = event.x - self.last_mouse[0]
        dy = event.y - self.last_mouse[1]
        self.last_mouse = (event.x, event.y)

        self.view.cx -= dx / self.view.zoom
        self.view.cy -= dy / self.view.zoom

        self.clamp_center()
        self.render()

    def end_pan(self, event):
        self.dragging = False

    def clamp_center(self):
        if self.view is None:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        half_w = w / (2 * self.view.zoom)
        half_h = h / (2 * self.view.zoom)

        if half_w >= self.source_width / 2:
            self.view.cx = self.source_width / 2
        else:
            self.view.cx = max(half_w, min(self.source_width - half_w, self.view.cx))

        if half_h >= self.source_height / 2:
            self.view.cy = self.source_height / 2
        else:
            self.view.cy = max(half_h, min(self.source_height - half_h, self.view.cy))

    def on_resize(self, event):
        size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if size == self._last_canvas_size:
            return
        self._last_canvas_size = size

        if self.source is not None and self.view is not None:
            self.render()

    def visible_source_box(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        left = self.view.cx - w / (2 * self.view.zoom)
        top = self.view.cy - h / (2 * self.view.zoom)
        right = self.view.cx + w / (2 * self.view.zoom)
        bottom = self.view.cy + h / (2 * self.view.zoom)

        return (
            max(0, int(math.floor(left))),
            max(0, int(math.floor(top))),
            min(self.source_width, int(math.ceil(right))),
            min(self.source_height, int(math.ceil(bottom))),
        )

    def choose_level_scale(self):
        """
        Choose a virtual resolution level.

        We want roughly 1 source pixel per screen pixel, but when zoomed out
        we can use a lower-resolution representation.

        This does not create the complete level. Only visible tiles are made.
        """
        z = self.view.zoom

        # source pixels per screen pixel = 1/z
        # Pick a power-of-two downsample where displayed pixels are near 1:1.
        if z >= 1:
            level = 0
        else:
            level = max(0, int(math.floor(math.log2(1 / z))))

        return level

    def tile_key(self, level, tx, ty):
        return (self.filename, level, tx, ty)

    def _enqueue_tile_request(self, key, level, box, generation, priority):
        if key in self.pending_tiles:
            return

        self.pending_tiles.add(key)
        heapq.heappush(
            self.request_heap,
            (priority, next(self._request_counter), key, level, box, generation),
        )

    def _dispatch_pending(self):
        while self.active_decodes < self.max_workers and self.request_heap:
            _, _, key, level, box, generation = heapq.heappop(self.request_heap)
            self.active_decodes += 1
            self.executor.submit(self._decode_tile_worker, key, level, box, generation)

    def _decode_tile_worker(self, key, level, box, generation):
        """
        Runs off the main thread so a cache-miss decode never stalls the UI.
        """
        image = None
        if generation == self.render_generation:
            try:
                scale = 2 ** level
                use_reduced = self.reduced_source is not None and (scale > 1 or self.reduced_source.covers_native)
                if use_reduced:
                    image = self.reduced_source.decode_region(level, box)
                else:
                    image = self._decode_tile_from_resident_source(level, box)
            except (OSError, ValueError):
                image = None

        self.tile_queue.put((key, generation, image))

    def _ensure_source_loaded(self):
        if self._source_loaded:
            return
        with self.source_lock:
            if not self._source_loaded:
                self.source.load()
                self._source_loaded = True

    def _decode_tile_from_resident_source(self, level, box):
        """
        Used whenever a reduced decode wouldn't help: formats with no
        draft() support at all (e.g. TIFF), and native-resolution (scale==1)
        requests for any format, since draft() is a no-op at scale 1
        regardless of format. self.source gets decoded once — eagerly at
        load time for the former, lazily here for the latter — and every
        tile just slices a cheap in-memory region from it instead of
        re-decoding the whole file again. The lock guards both that one-time
        load and concurrent reads from the decode threads.
        """
        self._ensure_source_loaded()

        scale = 2 ** level
        target_w = max(1, math.ceil((box[2] - box[0]) / scale))
        target_h = max(1, math.ceil((box[3] - box[1]) / scale))

        with self.source_lock:
            region = self.source.crop(box)

        if region.size != (target_w, target_h):
            region = region.resize((target_w, target_h), Image.Resampling.BOX)

        return region

    def _drain_tile_queue(self):
        completed = 0

        while True:
            try:
                key, generation, image = self.tile_queue.get_nowait()
            except queue.Empty:
                break

            completed += 1
            self.pending_tiles.discard(key)
            if image is not None:
                self.tile_cache.put(key, image)

        if completed:
            self.active_decodes -= completed
            self._dispatch_pending()

            # A tile can finish with image=None because its generation went
            # stale (superseded by a later pan/zoom) before it got decoded,
            # not because it errored. Its key is now free of pending_tiles,
            # but nothing has re-requested it for the current generation --
            # _enqueue_tile_request dedupes on that same set, so unless
            # something rescans the visible tiles, a tile dropped this way
            # stays missing indefinitely. Rescanning on every completion,
            # not just successful ones, is what catches that and re-enqueues it.
            self.refresh_display()

        self.root.after(15, self._drain_tile_queue)

    def render(self):
        if self.source is None or self.view is None:
            return

        # Cancel obsolete in-flight tile requests by incrementing the generation.
        self.render_generation += 1
        self.refresh_display()

    def refresh_display(self):
        # For the initial fit view, use the cheap preview immediately.
        if self.view.zoom <= self.fit_zoom * 1.15:
            self.render_preview()
        else:
            self.render_tiles(self.render_generation)

    def render_preview(self):
        self.canvas.delete("all")

        display_w = max(1, round(self.source_width * self.view.zoom))
        display_h = max(1, round(self.source_height * self.view.zoom))

        if (display_w, display_h) != self.preview.size:
            display_preview = self.preview.resize((display_w, display_h), Image.Resampling.BILINEAR)
        else:
            display_preview = self.preview

        self.preview_tk = ImageTk.PhotoImage(display_preview)

        x, y = self.canvas_from_source(self.source_width / 2, self.source_height / 2)

        self.canvas.create_image(x, y, image=self.preview_tk, anchor="center")

        percent = self.view.zoom / self.fit_zoom * 100
        self.status.config(
            text=f"{self.source_width:,} × {self.source_height:,}    "
                 f"Fit view ({percent:.0f}%)    "
                 f"RAM tiles: {len(self.tile_cache._cache)}"
        )

    def render_tiles(self, generation):
        self.canvas.delete("all")

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        level = self.choose_level_scale()
        scale = 2 ** level

        # Source region represented by each virtual tile.
        tile_source_size = TILE_SIZE * scale

        left, top, right, bottom = self.visible_source_box()

        tx0 = max(0, int(left // tile_source_size) - 1)
        ty0 = max(0, int(top // tile_source_size) - 1)
        tx1 = int((right - 1) // tile_source_size) + 1
        ty1 = int((bottom - 1) // tile_source_size) + 1

        # Draw visible tiles. A one-tile border gives us simple prefetching.
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                source_left = tx * tile_source_size
                source_top = ty * tile_source_size
                source_right = min(self.source_width, source_left + tile_source_size)
                source_bottom = min(self.source_height, source_top + tile_source_size)

                if source_left >= source_right or source_top >= source_bottom:
                    continue

                edge_left = w / 2 + (source_left - self.view.cx) * self.view.zoom
                edge_top = h / 2 + (source_top - self.view.cy) * self.view.zoom
                edge_right = w / 2 + (source_right - self.view.cx) * self.view.zoom
                edge_bottom = h / 2 + (source_bottom - self.view.cy) * self.view.zoom

                screen_x = round(edge_left)
                screen_y = round(edge_top)

                # Deriving width/height from the rounded shared edges (rather than
                # rounding tile.width * zoom * scale on its own) keeps neighboring
                # tiles pixel-aligned so no seam shows between them.
                screen_w = max(1, round(edge_right) - screen_x)
                screen_h = max(1, round(edge_bottom) - screen_y)

                key = self.tile_key(level, tx, ty)
                tile = self.tile_cache.get(key)

                clip = self._clip_to_canvas(screen_x, screen_y, screen_w, screen_h, w, h)

                if tile is not None:
                    if clip is not None:
                        crop_box = self._sub_crop_box(
                            (0, 0, tile.width, tile.height),
                            screen_x, screen_y, screen_w, screen_h, clip,
                        )
                        self._draw_patch(tile, crop_box, *clip)
                else:
                    ancestor = self._find_cached_ancestor(level, tx, ty)
                    if ancestor is not None:
                        if clip is not None:
                            self._draw_ancestor_fallback(
                                ancestor,
                                source_left, source_top, source_right, source_bottom,
                                screen_x, screen_y, screen_w, screen_h, clip,
                            )
                    else:
                        if clip is not None:
                            self._draw_preview_fallback(
                                source_left, source_top, source_right, source_bottom,
                                screen_x, screen_y, screen_w, screen_h, clip,
                            )

                    tile_center_x = (edge_left + edge_right) / 2
                    tile_center_y = (edge_top + edge_bottom) / 2
                    priority = math.hypot(tile_center_x - w / 2, tile_center_y - h / 2)

                    self._enqueue_tile_request(
                        key, level,
                        (source_left, source_top, source_right, source_bottom),
                        generation, priority,
                    )

        self._dispatch_pending()

        # Release references from old frames after the frame has been created.
        # This is separate from the actual decoded tile cache.
        if len(self._tk_tiles) > 200:
            self._tk_tiles = self._tk_tiles[-100:]

        percent = self.view.zoom / self.fit_zoom * 100

        self.status.config(
            text=f"{self.source_width:,} × {self.source_height:,}    "
                 f"Zoom: {percent:.1f}%    "
                 f"Level: 1/{scale}    "
                 f"Cached tiles: {len(self.tile_cache._cache)}"
        )

    def _clip_to_canvas(self, x, y, w, h, canvas_w, canvas_h):
        """
        Intersect a tile's screen rectangle with the canvas viewport.
        Whenever zoom > 1, a single native-resolution tile's screen size can
        be many times the canvas (e.g. a 512px tile at zoom 32 is 16384px
        on screen), so drawing it unclipped means resizing to that full
        unclipped size even though only a small fraction is ever visible --
        that unbounded resize cost is what makes panning slow at high zoom.
        Returns None if the tile has no visible overlap with the canvas.
        """
        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + w, canvas_w)
        y1 = min(y + h, canvas_h)
        if x0 >= x1 or y0 >= y1:
            return None
        return x0, y0, x1 - x0, y1 - y0

    def _sub_crop_box(self, crop_box, x, y, w, h, clip):
        """
        Narrow a crop_box (in the source image's own pixel coordinates,
        covering the full unclipped screen rect (x, y, w, h)) down to just
        the portion that overlaps the clipped rect, by linear interpolation.
        """
        clip_x, clip_y, clip_w, clip_h = clip
        left, top, right, bottom = crop_box
        return (
            left + (clip_x - x) / w * (right - left),
            top + (clip_y - y) / h * (bottom - top),
            left + (clip_x + clip_w - x) / w * (right - left),
            top + (clip_y + clip_h - y) / h * (bottom - top),
        )

    def _find_cached_ancestor(self, level, tx, ty, max_levels=10):
        """
        Walk up the level pyramid looking for an already-decoded coarser tile
        that covers the same source region as (level, tx, ty). Levels are
        nested two-to-one (a tile at level L is exactly one quadrant of its
        parent at level L+1), so tx/ty just keep halving.
        """
        for n in range(1, max_levels + 1):
            parent_level = level + n
            parent_tx = tx >> n
            parent_ty = ty >> n
            cached = self.tile_cache.get(self.tile_key(parent_level, parent_tx, parent_ty))
            if cached is not None:
                return cached, parent_level, parent_tx, parent_ty
        return None

    def _draw_patch(self, image, crop_box, x, y, w, h):
        left = max(0, int(math.floor(crop_box[0])))
        top = max(0, int(math.floor(crop_box[1])))
        right = min(image.width, max(left + 1, int(math.ceil(crop_box[2]))))
        bottom = min(image.height, max(top + 1, int(math.ceil(crop_box[3]))))

        patch = image.crop((left, top, right, bottom))
        if patch.size != (w, h):
            patch = patch.resize((w, h), Image.Resampling.BILINEAR)

        tk_patch = ImageTk.PhotoImage(patch)
        self.canvas.create_image(x, y, image=tk_patch, anchor="nw")
        self._tk_tiles.append(tk_patch)

    def _draw_ancestor_fallback(self, ancestor, source_left, source_top, source_right, source_bottom, x, y, w, h, clip):
        """
        Placeholder drawn from an already-cached coarser tile instead of the
        global preview — much sharper, since it's a decode of just this
        neighborhood rather than a downsample of the whole image.
        """
        image, parent_level, parent_tx, parent_ty = ancestor

        parent_scale = 2 ** parent_level
        parent_tile_source_size = TILE_SIZE * parent_scale
        parent_source_left = parent_tx * parent_tile_source_size
        parent_source_top = parent_ty * parent_tile_source_size

        crop_box = (
            (source_left - parent_source_left) / parent_scale,
            (source_top - parent_source_top) / parent_scale,
            (source_right - parent_source_left) / parent_scale,
            (source_bottom - parent_source_top) / parent_scale,
        )
        crop_box = self._sub_crop_box(crop_box, x, y, w, h, clip)
        self._draw_patch(image, crop_box, *clip)

    def _draw_preview_fallback(self, source_left, source_top, source_right, source_bottom, x, y, w, h, clip):
        """
        Last-resort placeholder for a tile that hasn't decoded yet and has no
        cached ancestor tile: an upscaled crop of the whole-image preview.
        """
        crop_box = (
            source_left / self.source_width * self.preview.width,
            source_top / self.source_height * self.preview.height,
            source_right / self.source_width * self.preview.width,
            source_bottom / self.source_height * self.preview.height,
        )
        crop_box = self._sub_crop_box(crop_box, x, y, w, h, clip)
        self._draw_patch(self.preview, crop_box, *clip)


def main():
    parser = argparse.ArgumentParser(
        description="Demand-driven large JPEG/TIFF image viewer"
    )
    parser.add_argument("image", help="Path to a JPEG or TIFF image")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        parser.error(f"File does not exist: {args.image}")

    root = tk.Tk()
    LargeImageViewer(root, args.image)
    root.mainloop()


if __name__ == "__main__":
    main()