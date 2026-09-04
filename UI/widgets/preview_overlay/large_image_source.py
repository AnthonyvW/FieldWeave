from __future__ import annotations

import math
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile
from PIL import Image

# image routinely exceeds Pillow's default decompression-bomb threshold
Image.MAX_IMAGE_PIXELS = None

TILE_SIZE = 512
# Generous relative to a single viewport's worth of tiles (typically a
# few dozen at most) — the point is headroom against eviction mid-zoom:
# every tile is the same TILE_SIZE regardless of pyramid level, and a
# continuous zoom gesture touches many levels in quick succession, so a
# tighter cap risked evicting a still-useful ancestor tile before the
# finer tile it was standing in for had even finished decoding.
CACHE_TILES = 512
PREVIEW_MAX = 1400
DEFAULT_MAX_WORKERS = max(2, (os.cpu_count() or 2) - 2)


class FrameSource(ABC):
    """
    Common interface ZoomPreviewOverlay crops/pans/zooms against, whether
    the underlying frame is a live camera frame already fully resident in
    memory (see zoom_preview.LiveFrameSource) or a possibly huge loaded
    image backed by LargeImageSource's tile cache.
    """

    @abstractmethod
    def dims(self) -> tuple[int, int]:
        """Return (height, width) in native-resolution pixels."""

    @abstractmethod
    def thumbnail(self) -> np.ndarray:
        """Return a small RGB array covering the whole frame, for the minimap."""

    @abstractmethod
    def region(self, box: tuple[int, int, int, int], step: int) -> np.ndarray:
        """
        Return an RGB array covering *box* (native-resolution source
        coordinates: left, top, right, bottom), downsampled by roughly
        *step*. Must never block on slow I/O — degrade to a coarser
        placeholder instead.
        """

    @abstractmethod
    def version(self) -> int:
        """
        Monotonically increasing counter that changes whenever a call to
        ``region`` could now return something different for the same
        arguments. A caller compares this against its own last-seen value
        to notice a background decode landed and its draw cache is stale.
        """


def _pil_to_array(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image)


class _ReducedSource:
    """Backend for decoding a region more cheaply than a full resident decode."""

    covers_native = False

    def decode_region(self, level: int, box: tuple[int, int, int, int]) -> np.ndarray:
        raise NotImplementedError

    def build_preview(self, preview_max: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _JpegDraftBackend(_ReducedSource):
    """
    JPEG's draft() gives a cheap reduced-resolution decode of the whole
    file. A JpegImageFile has a single internal tile covering the whole
    image, so crop()+load() on it always decodes the entire draft-scaled
    image regardless of the crop box — _level_cache decodes each level
    once and keeps it resident so every region request is a cheap crop of
    an already-decoded image instead of a fresh full decode.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._level_cache: dict[int, Image.Image] = {}
        self._level_cache_lock = threading.Lock()
        self._native_size_cache: tuple[int, int] | None = None

    def _get_level_image(self, level: int) -> Image.Image:
        cached = self._level_cache.get(level)
        if cached is not None:
            return cached

        with self._level_cache_lock:
            cached = self._level_cache.get(level)
            if cached is not None:
                return cached

            handle = Image.open(self.filename)
            original_size = handle.size
            scale = 2 ** level
            if scale > 1:
                handle.draft("RGB", (max(1, original_size[0] // scale), max(1, original_size[1] // scale)))
            if handle.mode != "RGB":
                handle = handle.convert("RGB")
            handle.load()
            self._level_cache[level] = handle
            return handle

    def _native_size(self) -> tuple[int, int]:
        if self._native_size_cache is None:
            with Image.open(self.filename) as handle:
                self._native_size_cache = handle.size
        return self._native_size_cache

    def decode_region(self, level: int, box: tuple[int, int, int, int]) -> np.ndarray:
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
        return _pil_to_array(region)

    def build_preview(self, preview_max: int) -> np.ndarray:
        preview = Image.open(self.filename)
        if preview.mode != "RGB":
            preview = preview.convert("RGB")
        preview.thumbnail((preview_max, preview_max), Image.Resampling.LANCZOS)
        return _pil_to_array(preview)

    def close(self) -> None:
        for image in self._level_cache.values():
            image.close()
        self._level_cache.clear()


class _PyramidTiffBackend(_ReducedSource):
    """
    Backend for TIFFs with real pyramid levels already stored in the file.
    Maps the requested power-of-two level to the closest available file
    level and reads directly from it — for tiled levels, only the TIFF
    tile segments a request overlaps get read and decoded.

    ``covers_native`` is True when level 0 is itself tiled, meaning even
    native-resolution requests can be served straight from the file
    without ever decoding a full resident copy.
    """

    def __init__(self, tiff_file: tifffile.TiffFile, source_width: int, source_height: int) -> None:
        self._tf = tiff_file
        self.source_width = source_width
        self.source_height = source_height
        # Serializes filehandle seek+read across decode threads — decode()
        # itself (the CPU-bound part) runs outside this lock.
        self._file_lock = threading.Lock()
        self._whole_level_cache: dict[int, np.ndarray] = {}
        self._whole_level_cache_lock = threading.Lock()
        self._levels = self._build_level_table()
        self.covers_native = self._compute_covers_native()

    def _build_level_table(self) -> list[dict]:
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

    def _compute_covers_native(self) -> bool:
        level0 = next(
            (lv for lv in self._levels if lv["width"] == self.source_width and lv["height"] == self.source_height),
            None,
        )
        if level0 is None:
            return False
        return self._tf.pages[level0["page_index"]].is_tiled

    def _pick_level(self, requested_scale: float) -> dict | None:
        chosen = None
        for lv in self._levels:
            if lv["scale_x"] <= requested_scale:
                chosen = lv
            else:
                break
        return chosen or (self._levels[-1] if self._levels else None)

    def _read_segment(self, page, index: int, offsets, counts) -> np.ndarray | None:
        with self._file_lock:
            self._tf.filehandle.seek(offsets[index])
            data = self._tf.filehandle.read(counts[index])
        seg, _pos, _shape = page.decode(data, index, jpegtables=page.jpegtables)
        return None if seg is None else seg[0]

    def _decode_tiled_region(self, page, region_box: tuple[int, int, int, int]) -> np.ndarray | None:
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

    def _get_whole_level(self, level_index: int, page) -> np.ndarray:
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

    def _decode_level_box(self, file_level: dict, region_box: tuple[int, int, int, int]) -> np.ndarray:
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

        if arr is None or arr.shape[:2] != (full_h, full_w):
            canvas = np.zeros((max(full_h, 1), max(full_w, 1), samples), dtype=np.uint8)
            if arr is not None and arr.size:
                off_y = clipped[1] - req_top
                off_x = clipped[0] - req_left
                canvas[off_y:off_y + arr.shape[0], off_x:off_x + arr.shape[1]] = arr
            arr = canvas

        if samples >= 3:
            arr = arr[:, :, :3]
        else:
            arr = np.repeat(arr[:, :, :1], 3, axis=2)
        return arr

    def decode_region(self, level: int, box: tuple[int, int, int, int]) -> np.ndarray:
        scale = 2 ** level
        file_level = self._pick_level(scale)
        if file_level is None:
            raise ValueError("no pyramid levels available")

        region_box = (
            round(box[0] / file_level["scale_x"]), round(box[1] / file_level["scale_y"]),
            round(box[2] / file_level["scale_x"]), round(box[3] / file_level["scale_y"]),
        )
        arr = self._decode_level_box(file_level, region_box)

        target_w = max(1, math.ceil((box[2] - box[0]) / scale))
        target_h = max(1, math.ceil((box[3] - box[1]) / scale))
        if arr.shape[:2] != (target_h, target_w):
            arr = _pil_to_array(Image.fromarray(arr).resize((target_w, target_h), Image.Resampling.BOX))
        return arr

    def build_preview(self, preview_max: int) -> np.ndarray:
        """
        Pick the stored level whose resolution best matches preview_max —
        the finest one that's still no more detailed than needed, via the
        same _pick_level used for ordinary tile requests — rather than
        always the single coarsest stored level regardless of how small
        that actually is. A pyramid with many levels can have a coarsest
        level far smaller than preview_max, which produced an
        unnecessarily blurry initial preview and a visible jump in
        sharpness the moment real zoomed tile requests picked a properly-
        matched, finer level instead.
        """
        if not self._levels:
            raise ValueError("no pyramid levels available")
        requested_scale = max(self.source_width, self.source_height) / preview_max
        file_level = self._pick_level(requested_scale)
        arr = self._decode_level_box(file_level, (0, 0, file_level["width"], file_level["height"]))
        image = Image.fromarray(arr)
        image.thumbnail((preview_max, preview_max), Image.Resampling.LANCZOS)
        return _pil_to_array(image)

    def close(self) -> None:
        self._tf.close()


def _detect_reduced_source(
    filename: str, source_format: str | None, source_width: int, source_height: int,
) -> _ReducedSource | None:
    """
    Choose the cheaper-than-native decode backend for this file, or None
    if there isn't one (a flat, non-pyramid TIFF, or any other format —
    either way the resident-decode fallback still works, just without a
    zoomed-out shortcut).
    """
    if source_format in ("JPEG", "MPO"):
        return _JpegDraftBackend(filename)

    if source_format != "TIFF":
        return None

    tf = None
    try:
        tf = tifffile.TiffFile(filename)
        if tf.series[0].is_pyramidal:
            return _PyramidTiffBackend(tf, source_width, source_height)
        tf.close()
    except (OSError, ValueError, KeyError):
        # A corrupt or unusual pyramid tag just means treating the file as
        # a flat TIFF via the resident-decode fallback below.
        if tf is not None:
            tf.close()
    return None


class LargeImageSource(FrameSource):
    """
    Demand-driven, tile-cached access to a possibly huge (many-thousand-
    pixel-per-side) image file, ported from the standalone large-image
    viewer prototype (main.py).

    Never holds a decoded copy of the full image — only a small preview
    plus whichever tiles have actually been requested. A JPEG gets
    Pillow's cheap reduced-resolution draft() decode for zoomed-out
    levels; a pyramid TIFF reads directly from its stored lower-resolution
    levels; anything else falls back to a single resident decode kept in
    memory, since there's no cheaper option for a flat file.

    All state here is plain data guarded by ``_cache_lock`` — this class
    runs its own background decode threads but never touches a widget,
    signal, or setting itself. ``version`` increments whenever a tile
    finishes decoding; a caller compares it against its own last-seen
    value to notice a background decode landed and knows to repaint —
    see ZoomPreviewOverlay's draw cache and CameraPreview's poll timer.
    """

    def __init__(self, filename: str, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        self.filename = filename
        self.source_width = 0
        self.source_height = 0
        self.preview: np.ndarray | None = None
        self._version = 0

        self._reduced_source: _ReducedSource | None = None
        self._resident: Image.Image | None = None
        self._resident_loaded = False
        self._resident_lock = threading.Lock()

        self._tile_cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._pending: set[tuple[int, int, int]] = set()
        self._cache_lock = threading.Lock()

        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tile-decode")

    def open(self) -> bool:
        """
        Detect the file's format and build the initial preview. The
        caller is responsible for running this off the main thread —
        see CaptureControlWidget._load_image_routine.

        Wrapped as a whole: decoding an arbitrary user-supplied file can
        fail in more ways than just a bad path (truncated data, a format
        Pillow/tifffile mishandles, and so on), and any of those should
        report failure rather than crash the background thread — which
        would otherwise leave the caller waiting on a result that never
        arrives.
        """
        try:
            handle = Image.open(self.filename)
            if handle.mode not in ("RGB", "RGBA"):
                handle = handle.convert("RGB")
            self.source_width, self.source_height = handle.size

            self._reduced_source = _detect_reduced_source(
                self.filename, handle.format, self.source_width, self.source_height,
            )
            # Kept open (but not decoded — see _load_resident) even when a
            # reduced_source backend exists: that backend only ever helps
            # at scale > 1 (or, for a pyramid TIFF whose level 0 is
            # itself tiled, covers_native — see PyramidTiffBackend).
            # Anything else at native zoom still needs this as the
            # fallback, e.g. a plain JPEG has no cheaper native-resolution
            # decode than Pillow's own crop.
            self._resident = handle

            if self._reduced_source is not None:
                self.preview = self._reduced_source.build_preview(PREVIEW_MAX)
            else:
                self._load_resident()
                self.preview = self._build_preview_from_resident()
        except Exception:
            return False

        return True

    def _load_resident(self) -> None:
        if self._resident_loaded:
            return
        with self._resident_lock:
            if not self._resident_loaded:
                self._resident.load()
                self._resident_loaded = True

    def _build_preview_from_resident(self) -> np.ndarray:
        ratio = min(PREVIEW_MAX / self.source_width, PREVIEW_MAX / self.source_height, 1.0)
        preview_w = max(1, round(self.source_width * ratio))
        preview_h = max(1, round(self.source_height * ratio))
        with self._resident_lock:
            resized = self._resident.resize((preview_w, preview_h), Image.Resampling.LANCZOS)
        return _pil_to_array(resized)

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._reduced_source is not None:
            self._reduced_source.close()
        if self._resident is not None:
            self._resident.close()

    # ------------------------------------------------------------------
    # FrameSource interface
    # ------------------------------------------------------------------

    def dims(self) -> tuple[int, int]:
        return self.source_height, self.source_width

    @property
    def is_pyramid(self) -> bool:
        """Whether this source decodes on demand from a reduced-resolution backend (a real pyramid TIFF, or a JPEG's cheap draft decode) rather than a plain flat file — reading its full native-resolution region (see region()) can mean decoding a multi-gigapixel image into memory, so callers use this to withhold operations that would do that (e.g. a "full-res" export) for a source like this."""
        return self._reduced_source is not None

    def thumbnail(self) -> np.ndarray:
        return self.preview if self.preview is not None else np.zeros((1, 1, 3), dtype=np.uint8)

    def version(self) -> int:
        return self._version

    def region(self, box: tuple[int, int, int, int], step: int) -> np.ndarray:
        level = max(0, int(math.floor(math.log2(max(1, step)))))
        left = max(0, box[0])
        top = max(0, box[1])
        right = min(self.source_width, box[2])
        bottom = min(self.source_height, box[3])

        scale = 2 ** level
        # Every tile's output position is derived from this same function
        # of its absolute source coordinate, rather than each tile sizing
        # itself independently — two adjacent tiles' shared source-space
        # edge then always rounds to the same output column, so there's
        # no seam or gap between them. See _composite_tile.
        def edge_x(source_x: int) -> int:
            return round((source_x - left) / scale)

        def edge_y(source_y: int) -> int:
            return round((source_y - top) / scale)

        out_w = max(1, edge_x(right))
        out_h = max(1, edge_y(bottom))
        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if right <= left or bottom <= top:
            return out

        tile_source_size = TILE_SIZE * scale
        tx0 = int(left // tile_source_size)
        ty0 = int(top // tile_source_size)
        tx1 = int((right - 1) // tile_source_size)
        ty1 = int((bottom - 1) // tile_source_size)

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                self._composite_tile(out, level, tx, ty, edge_x, edge_y)

        return out

    # ------------------------------------------------------------------
    # Tile cache
    # ------------------------------------------------------------------

    def get_tile(self, level: int, tx: int, ty: int) -> np.ndarray | None:
        key = (level, tx, ty)
        with self._cache_lock:
            tile = self._tile_cache.get(key)
            if tile is not None:
                self._tile_cache.move_to_end(key)
            return tile

    def get_ancestor_tile(
        self, level: int, tx: int, ty: int, max_levels: int = 10,
    ) -> tuple[np.ndarray, int, int, int] | None:
        """
        Walk up the pyramid for an already-cached coarser tile covering
        the same region — used as a placeholder while the exact tile is
        still decoding. Levels nest two-to-one, so tx/ty just keep halving.
        """
        for n in range(1, max_levels + 1):
            parent_level = level + n
            tile = self.get_tile(parent_level, tx >> n, ty >> n)
            if tile is not None:
                return tile, parent_level, tx >> n, ty >> n
        return None

    def request_tile(self, level: int, tx: int, ty: int) -> None:
        """Kick off a background decode if this tile isn't already cached or in flight."""
        key = (level, tx, ty)
        with self._cache_lock:
            if key in self._tile_cache or key in self._pending:
                return
            self._pending.add(key)
        self._executor.submit(self._decode_tile, key)

    def _decode_tile(self, key: tuple[int, int, int]) -> None:
        """
        Runs on a background thread. Mutates only the tile cache and
        version — see class docstring.

        A decode failure here must still clear *key* from ``_pending`` —
        an unhandled exception raised inside a ThreadPoolExecutor-
        submitted call is silently dropped by concurrent.futures rather
        than propagating anywhere visible, which would otherwise leave
        this key stuck in ``_pending`` forever: request_tile's dedup
        check would then skip it on every future call, and that tile
        could never be decoded for the rest of the session.
        """
        level, tx, ty = key
        scale = 2 ** level
        tile_source_size = TILE_SIZE * scale
        left = tx * tile_source_size
        top = ty * tile_source_size
        right = min(self.source_width, left + tile_source_size)
        bottom = min(self.source_height, top + tile_source_size)

        array = None
        try:
            if not self._closed and right > left and bottom > top:
                box = (left, top, right, bottom)
                use_reduced = self._reduced_source is not None and (scale > 1 or self._reduced_source.covers_native)
                array = (
                    self._reduced_source.decode_region(level, box) if use_reduced
                    else self._decode_from_resident(box, level)
                )
        except Exception:
            array = None

        with self._cache_lock:
            self._pending.discard(key)
            if array is not None:
                self._tile_cache[key] = array
                self._tile_cache.move_to_end(key)
                while len(self._tile_cache) > CACHE_TILES:
                    self._tile_cache.popitem(last=False)
                self._version += 1

    def _decode_from_resident(self, box: tuple[int, int, int, int], level: int) -> np.ndarray | None:
        """
        Used for anything with no cheaper reduced-resolution decode path
        (a flat, non-pyramid TIFF) or for native-resolution requests on a
        format that does have one. Crops the resident image at native
        resolution, then — critically, at a zoomed-out level — downsamples
        that crop to the tile's actual target size, matching what both
        _JpegDraftBackend and _PyramidTiffBackend already do for their own
        decode paths. Skipping this was the actual perf bug: a level-3
        tile's native-resolution box is 8x8 = 64 times the pixel count of
        the ~TILE_SIZE output it should be, so every zoomed-out tile on a
        flat image was being cropped, cached, and later resized down from
        far more data than it ever needed to carry.
        """
        if self._resident is None:
            return None
        self._load_resident()
        with self._resident_lock:
            region = self._resident.crop(box)

        scale = 2 ** level
        target_w = max(1, math.ceil((box[2] - box[0]) / scale))
        target_h = max(1, math.ceil((box[3] - box[1]) / scale))
        if region.size != (target_w, target_h):
            region = region.resize((target_w, target_h), Image.Resampling.BOX)
        return _pil_to_array(region)

    def _composite_tile(
        self, out: np.ndarray, level: int, tx: int, ty: int, edge_x: Callable[[int], int], edge_y: Callable[[int], int],
    ) -> None:
        scale = 2 ** level
        tile_source_size = TILE_SIZE * scale
        tile_left = tx * tile_source_size
        tile_top = ty * tile_source_size
        tile_right = min(self.source_width, tile_left + tile_source_size)
        tile_bottom = min(self.source_height, tile_top + tile_source_size)
        if tile_left >= tile_right or tile_top >= tile_bottom:
            return

        self.request_tile(level, tx, ty)

        # This tile's slot in *out*, from the same edge() function every
        # other tile uses — not this patch's own natural decoded size —
        # so it always exactly abuts its neighbors regardless of how any
        # particular backend happened to round its own crop/resize.
        dst_left, dst_top = edge_x(tile_left), edge_y(tile_top)
        dst_w = edge_x(tile_right) - dst_left
        dst_h = edge_y(tile_bottom) - dst_top
        if dst_w <= 0 or dst_h <= 0:
            return

        patch = self._resolve_patch(level, tx, ty, tile_left, tile_top, tile_right, tile_bottom)
        if patch is None:
            return
        if patch.shape[:2] != (dst_h, dst_w):
            patch = _pil_to_array(Image.fromarray(patch).resize((dst_w, dst_h), Image.Resampling.BILINEAR))

        src_x0 = max(0, -dst_left)
        src_y0 = max(0, -dst_top)
        dst_x0 = max(0, dst_left)
        dst_y0 = max(0, dst_top)
        dst_x1 = min(out.shape[1], dst_left + dst_w)
        dst_y1 = min(out.shape[0], dst_top + dst_h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = patch[src_y0:src_y1, src_x0:src_x1]

    def _resolve_patch(
        self, level: int, tx: int, ty: int, tile_left: int, tile_top: int, tile_right: int, tile_bottom: int,
    ) -> np.ndarray | None:
        """Best tile-sized patch available right now: the exact tile, else a cropped ancestor, else the preview."""
        exact = self.get_tile(level, tx, ty)
        if exact is not None:
            return exact

        ancestor = self.get_ancestor_tile(level, tx, ty)
        if ancestor is not None:
            array, parent_level, parent_tx, parent_ty = ancestor
            parent_scale = 2 ** parent_level
            parent_tile_source_size = TILE_SIZE * parent_scale
            parent_left = parent_tx * parent_tile_source_size
            parent_top = parent_ty * parent_tile_source_size
            crop_left = (tile_left - parent_left) // parent_scale
            crop_top = (tile_top - parent_top) // parent_scale
            ph, pw = array.shape[:2]
            crop_right = min(pw, crop_left + max(1, (tile_right - tile_left) // parent_scale))
            crop_bottom = min(ph, crop_top + max(1, (tile_bottom - tile_top) // parent_scale))
            if crop_right <= crop_left or crop_bottom <= crop_top:
                return None
            return array[crop_top:crop_bottom, crop_left:crop_right]

        if self.preview is None or self.source_width == 0 or self.source_height == 0:
            return None
        ph, pw = self.preview.shape[:2]
        crop_left = int(tile_left / self.source_width * pw)
        crop_top = int(tile_top / self.source_height * ph)
        crop_right = min(pw, max(crop_left + 1, int(tile_right / self.source_width * pw)))
        crop_bottom = min(ph, max(crop_top + 1, int(tile_bottom / self.source_height * ph)))
        return self.preview[crop_top:crop_bottom, crop_left:crop_right]