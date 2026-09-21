from __future__ import print_function
import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np

DEFAULT_CONFIG = {
    'max_features': 500,
    'scale_factor': 0.25,
    'flann_checks': 12,
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

STACKED_DIRNAME = 'focus_stacked'
CALIBRATION_DIRNAME = 'calibration_slide'
CALIBRATION_STEM = 'calibration_slide'


def natural_sort_key(path):
    """Sort '...img2.jpg' before '...img10.jpg' by splitting off numeric runs."""
    name = os.path.basename(path)
    return [int(chunk) if chunk.isdigit() else chunk.lower()
            for chunk in re.split(r'(\d+)', name)]


def collect_images(inputs, no_sort=False):
    """Resolve CLI input arguments (a directory, files, or globs) to an
    ordered list of image paths."""
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for entry in os.listdir(item):
                if entry.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(os.path.join(item, entry))
        elif any(ch in item for ch in '*?[]'):
            paths.extend(glob.glob(item))
        else:
            paths.append(item)

    if no_sort:
        # Preserve the order given on the command line / by the directory
        # listing, just dropping repeats (keeping the first occurrence).
        seen = set()
        deduped = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return deduped

    return sorted(set(paths), key=natural_sort_key)


def resolve_tree_core(inputs):
    """Return (tree_core_folder, image_inputs) when a single tree core folder
    was passed, else (None, inputs) so files/globs keep working as before."""
    if len(inputs) != 1 or not os.path.isdir(inputs[0]):
        return None, inputs

    root = os.path.abspath(inputs[0])
    stacked = os.path.join(root, STACKED_DIRNAME)
    if not os.path.isdir(stacked):
        return None, inputs

    return root, [stacked]


def find_calibration_slide(tree_core_folder):
    """Return (slide_path, already_horizontal) for the tree core's calibration
    slide, or (None, False) when the folder or the image is absent."""
    cal_dir = os.path.join(tree_core_folder, CALIBRATION_DIRNAME)
    if not os.path.isdir(cal_dir):
        return None, False

    entries = sorted(os.listdir(cal_dir))
    slide_path = None
    for entry in entries:
        stem, ext = os.path.splitext(entry)
        if stem.lower() == CALIBRATION_STEM and ext.lower() in IMAGE_EXTENSIONS:
            slide_path = os.path.join(cal_dir, entry)
            break

    if slide_path is None:
        return None, False

    # DPI.txt is only written once the slide has been measured, a pass that
    # leaves the image horizontal regardless of how the core was scanned.
    already_horizontal = any(e.lower() == 'dpi.txt' for e in entries)
    return slide_path, already_horizontal


def prepend_calibration_slide(composite, slide):
    """Butt the calibration slide against the left edge of the stitched core,
    vertically centred, without overlapping it."""
    height = max(composite.shape[0], slide.shape[0])
    width = slide.shape[1] + composite.shape[1]
    canvas = np.zeros((height, width, 3), np.uint8)

    slide_y = (height - slide.shape[0]) // 2
    canvas[slide_y:slide_y + slide.shape[0], 0:slide.shape[1]] = slide

    comp_y = (height - composite.shape[0]) // 2
    canvas[comp_y:comp_y + composite.shape[0], slide.shape[1]:width] = composite

    return canvas


class RegistrationError(RuntimeError):
    """A pair carries too little detail in its overlap to align. Past either
    end of a core that means the out-of-focus tray, not a stitching fault."""


class Stitcher:
    """Direct port of stitcher.py's Stitcher class: SIFT + FLANN registration
    in the overlap region, then a hard-seam (no blend, no warp) composite."""

    def __init__(self, overlap, config=None, verbose=True):
        self.overlap = overlap
        self.config = config or DEFAULT_CONFIG
        self.maxOffset = 0
        self.verbose = verbose
        self.runs = []
        self.stitched_range = None

    def log(self, message):
        if self.verbose:
            print(message)

    def warn(self, message):
        print('  Warning: {}'.format(message), file=sys.stderr)

    def load_image(self, path, vertical_core):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError('Could not read image: {}'.format(path))
        if vertical_core:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def calculate_offset(self, img1, img2, enable_mask):
        cfg = self.config

        # Calculate rough overlap in pixels
        overlap_px = img2.shape[1] * self.overlap

        # Convert images to grayscale and reduce size to scale_factor
        i1 = cv2.cvtColor(
            cv2.resize(img1[:, -int(overlap_px):, :], (0, 0),
                       fx=cfg['scale_factor'], fy=cfg['scale_factor']),
            cv2.COLOR_BGR2GRAY)
        i2 = cv2.cvtColor(
            cv2.resize(img2[:, :int(overlap_px), :], (0, 0),
                       fx=cfg['scale_factor'], fy=cfg['scale_factor']),
            cv2.COLOR_BGR2GRAY)

        if enable_mask:
            height, width = i1.shape[:2]
            mask = np.zeros(i1.shape[:2], np.uint8)
            rounded = round(height / 4)
            mask[rounded:height - rounded, 0:width] = 255
        else:
            mask = None

        # Find SIFT keypoints and descriptors
        sift = cv2.SIFT_create(nfeatures=cfg['max_features'])
        self.log('\t- Finding keypoints and descriptors for image 1')
        kp1, des1 = sift.detectAndCompute(i1, mask)
        self.log('\t- Image 1 overlap: {} keypoints, {} descriptors'.format(
            len(kp1), 0 if des1 is None else len(des1)))
        self.log('\t- Finding keypoints and descriptors for image 2')
        kp2, des2 = sift.detectAndCompute(i2, mask)
        self.log('\t- Image 2 overlap: {} keypoints, {} descriptors'.format(
            len(kp2), 0 if des2 is None else len(des2)))

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            raise RegistrationError(
                'not enough SIFT keypoints in the overlap region ({} and {} '
                'descriptors). Try a larger --overlap, disable --mask, or '
                'increase --max-features'.format(
                    0 if des1 is None else len(des1),
                    0 if des2 is None else len(des2)))

        # Use FLANN to determine matches
        self.log('\t- Finding matches')
        flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5},
                                       {'checks': cfg['flann_checks']})
        matches = flann.knnMatch(des1, des2, k=2)

        # Limit to reasonable matches
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if not good_matches:
            raise RegistrationError(
                'no SIFT match survived the ratio test ({} and {} descriptors, '
                '{} raw matches)'.format(len(des1), len(des2), len(matches)))

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # We're not doing any robust analyses of outliers, so let's just
        # take the median and see how it works
        x_offset = int(np.median([elem[0][0] for elem in np.subtract(src_pts, dst_pts)]))
        y_offset = int(np.median([elem[0][1] for elem in np.subtract(src_pts, dst_pts)]))

        # Rescale offset for original size and return
        self.log('\t- X Offset found: {} px'.format(x_offset * (1 / cfg['scale_factor'])))
        self.log('\t- Y Offset found: {} px'.format(y_offset * (1 / cfg['scale_factor'])))
        return (x_offset * (1 / cfg['scale_factor']), y_offset * (1 / cfg['scale_factor']))

    def stitch_images(self, img1, img2, enable_mask):
        # Find the offset between the images and calculate where the seam lies
        x_offset, y_offset = self.calculate_offset(img1, img2, enable_mask)

        # We want to seam the images such that we crop the left and right
        # equally, so use roughly half the overlap between them.
        x_seam = int(img1.shape[1] - (img2.shape[1] * self.overlap * .5) + x_offset)

        # How much of the new image should we crop out?
        partial_image = int(img2.shape[1] * self.overlap * .5)

        if x_seam < 0 or x_seam > img1.shape[1]:
            raise RuntimeError(
                'Computed seam position ({} px) falls outside image 1\'s width '
                '({} px) -- SIFT measured an x offset of {:.1f}px, which is not '
                'consistent with --overlap {}. This usually means either: the '
                'wrong stitching axis is being used (pass --vertical-core if the '
                'core was scanned top-to-bottom rather than left-to-right), the '
                '--overlap value does not match the images\' real overlap, or '
                'this pair matched on the wrong content (try --mask, or increase '
                '--max-features).'.format(x_seam, img1.shape[1], x_offset, self.overlap))

        self.maxOffset = max(self.maxOffset, y_offset)

        # Create the composite image and return
        width = x_seam + (img2.shape[1] - partial_image)
        height = img2.shape[0] + abs(int(self.maxOffset))

        if y_offset < 0.0:
            comp_img = np.zeros((height + int(abs(y_offset)), width, 3), np.uint8)
            self.maxOffset += int(abs(y_offset))
            comp_img[int(abs(y_offset)):img1.shape[0] + int(abs(y_offset)), 0:x_seam] = \
                img1[0:img1.shape[0], 0:x_seam]
            comp_img[0:img2.shape[0], x_seam:(x_seam + img2.shape[1] - partial_image)] = \
                img2[0:img2.shape[0], partial_image:img2.shape[1]]
        else:
            comp_img = np.zeros((height, width, 3), np.uint8)
            comp_img[0:img1.shape[0], 0:x_seam] = img1[0:img1.shape[0], 0:x_seam]
            comp_img[int(y_offset):img2.shape[0] + int(y_offset),
                     x_seam:(x_seam + img2.shape[1] - partial_image)] = \
                img2[0:img2.shape[0], partial_image:img2.shape[1]]

        return comp_img

    def keep_longest_run(self, best, composite, first, last):
        """Close off the run of images just stitched and keep whichever of it
        and *best* covers more images.  Only the winner's composite is retained,
        since a full-core composite runs to hundreds of megabytes."""
        if composite is None:
            return best

        self.runs.append((first, last))
        run = {'first': first, 'last': last,
               'offset': self.maxOffset, 'composite': composite}
        if best is None or last - first > best['last'] - best['first']:
            return run
        return best

    def stitch_sequence(self, image_paths, enable_mask, vertical_core, crop):
        """Stitch the longest run of consecutive images that align with each
        other.  Pairs that cannot be registered -- the out-of-focus tray past
        either end of the core, or a dropout mid-core -- end the current run and
        start a new one, and the longest run wins."""
        self.runs = []
        names = [os.path.basename(p) for p in image_paths]

        best = None
        composite = None
        first = 0
        index = 0
        self.maxOffset = 0

        while index + 1 < len(image_paths):
            left = composite
            if left is None:
                left = self.load_image(image_paths[index], vertical_core)
            right = self.load_image(image_paths[index + 1], vertical_core)

            self.log('Stitching images {} and {} of {}: {} + {}'.format(
                index + 1, index + 2, len(image_paths),
                image_paths[index], image_paths[index + 1]))
            try:
                stitched = self.stitch_images(left, right, enable_mask)
            except RegistrationError as e:
                self.warn('Could not align image {} ({}) with image {} ({}): {}.'.format(
                    index + 1, names[index], index + 2, names[index + 1], e))
                best = self.keep_longest_run(best, composite, first, index)
                if composite is not None:
                    self.log('  Run of images {}-{} ends here; starting a new run '
                             'at image {}'.format(first + 1, index + 1, index + 2))
                composite = None
                self.maxOffset = 0
                first = index + 1
                index += 1
                continue
            except Exception as e:
                self.log('  Error stitching {} and {}: {}'.format(
                    image_paths[index], image_paths[index + 1], e))
                raise

            composite = stitched
            index += 1
            self.log('  Composite size so far: {}'.format(composite.shape))

        best = self.keep_longest_run(best, composite, first, index)
        if best is None:
            raise RuntimeError(
                'No pair of images could be aligned -- every overlap region was '
                'too featureless to register.')

        composite = best['composite']
        self.maxOffset = best['offset']
        self.stitched_range = (best['first'], best['last'])

        kept = best['last'] - best['first'] + 1
        if kept < len(image_paths):
            print('Aligned runs found:')
            for run_first, run_last in self.runs:
                print('  images {}-{} ({}-{}), {} image(s){}'.format(
                    run_first + 1, run_last + 1, names[run_first], names[run_last],
                    run_last - run_first + 1,
                    '  <- stitched' if (run_first, run_last) == self.stitched_range else ''))

            unaligned = [names[i] for i in range(len(image_paths))
                         if not any(f <= i <= l for f, l in self.runs)]
            if unaligned:
                self.warn('Dropped {} image(s) that aligned with nothing: {}.'.format(
                    len(unaligned), ', '.join(unaligned)))

            others = [r for r in self.runs if r != self.stitched_range]
            if others:
                self.warn('Dropped {} shorter run(s) that did align, so a section of '
                          'the core is missing from the output: {}.'.format(
                              len(others),
                              '; '.join('images {}-{}'.format(f + 1, l + 1)
                                        for f, l in others)))

            print('Stitched images {}-{} ({} of {} images)'.format(
                best['first'] + 1, best['last'] + 1, kept, len(image_paths)))

        if crop and self.maxOffset:
            self.log('Cropping composite by max Y drift ({} px)'.format(self.maxOffset))
            composite = composite[int(self.maxOffset):(composite.shape[0] - int(self.maxOffset)),
                                   0:composite.shape[1]]

        return composite


def main():
    parser = argparse.ArgumentParser(
        description='Stitch a sequence of overlapping images using '
                    'SIFT + FLANN alignment and hard-seam compositing.')
    parser.add_argument('inputs', nargs='+',
                        help='A tree core folder containing a "focus_stacked" '
                             'subfolder, a directory of images, or an explicit list '
                             'of image files/globs, in left-to-right scan order.')
    parser.add_argument('-o', '--output',
                        help='Path to write the stitched image to (default: '
                             '"<tree core name>.tiff" inside the tree core folder, '
                             'or "output.jpg" inside a plain input folder).')
    parser.add_argument('--calibration-slide', action='store_true',
                        help='Prepend the calibration slide image found in the tree '
                             'core folder\'s "calibration_slide" subfolder to the '
                             'left edge of the stitched core.')
    parser.add_argument('--overlap', type=float, default=0.35,
                        help='Expected overlap between adjacent images, as a fraction '
                             'of image width (default: 0.35).')
    parser.add_argument('--mask', action='store_true',
                        help='Restrict SIFT keypoint search to the vertical-center band '
                             'of the overlap region (ignores the top/bottom quarters).')
    parser.add_argument('--vertical-core', action='store_true',
                        help='Rotate each image 90° before stitching, for cores scanned '
                             'top-to-bottom instead of left-to-right.')
    parser.add_argument('--no-crop', dest='crop', action='store_false',
                        help='Skip cropping the final composite to remove the ragged '
                             'top/bottom edge left by vertical drift (cropped by default).')
    parser.add_argument('--preview', metavar='PATH',
                        help='Also write a scaled-down (max 1000px) preview to this path.')
    parser.add_argument('--no-sort', action='store_true',
                        help='Use the input files in the order given / directory listing '
                             'order instead of natural (numeric-aware) filename sort.')
    parser.add_argument('--reverse', action='store_true',
                        help='Process the images in the opposite order (right-to-left '
                             'instead of left-to-right).')
    parser.add_argument('--max-features', type=int, default=DEFAULT_CONFIG['max_features'],
                        help='SIFT nfeatures cap per image (default: %(default)s).')
    parser.add_argument('--scale-factor', type=float, default=DEFAULT_CONFIG['scale_factor'],
                        help='Downscale factor applied to the overlap region before SIFT '
                             '(default: %(default)s).')
    parser.add_argument('--flann-checks', type=int, default=DEFAULT_CONFIG['flann_checks'],
                        help='FLANN matcher "checks" parameter (default: %(default)s).')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress per-pair progress logging.')
    args = parser.parse_args()

    tree_core_folder, image_inputs = resolve_tree_core(args.inputs)
    if tree_core_folder is not None:
        print('Tree core folder: {}'.format(tree_core_folder))
    elif args.calibration_slide:
        parser.error('--calibration-slide requires a tree core folder containing a '
                     '"{}" subfolder.'.format(STACKED_DIRNAME))

    image_paths = collect_images(image_inputs, no_sort=args.no_sort)
    if len(image_paths) < 2:
        parser.error('Need at least 2 images to stitch, found {}.'.format(len(image_paths)))

    if args.reverse:
        image_paths = list(reversed(image_paths))

    if args.output:
        output_path = args.output
    elif tree_core_folder is not None:
        output_path = os.path.join(
            tree_core_folder, '{}.tiff'.format(os.path.basename(tree_core_folder)))
    else:
        first_input = os.path.abspath(args.inputs[0])
        input_dir = first_input if os.path.isdir(first_input) else os.path.dirname(first_input)
        output_path = os.path.join(input_dir, 'output.jpg')

    print('Stitching {} images{}:'.format(
        len(image_paths), ' (reversed)' if args.reverse else ''))
    for p in image_paths:
        print('  {}'.format(p))

    config = {
        'max_features': args.max_features,
        'scale_factor': args.scale_factor,
        'flann_checks': args.flann_checks,
    }

    stitcher = Stitcher(args.overlap, config=config, verbose=not args.quiet)
    try:
        composite = stitcher.stitch_sequence(
            image_paths, enable_mask=args.mask, vertical_core=args.vertical_core,
            crop=args.crop)
    except Exception as e:
        print('Stitching failed: {}'.format(e), file=sys.stderr)
        return 1

    if args.calibration_slide:
        slide_path, already_horizontal = find_calibration_slide(tree_core_folder)
        if slide_path is None:
            print('No calibration slide found in {}; skipping.'.format(
                os.path.join(tree_core_folder, CALIBRATION_DIRNAME)), file=sys.stderr)
        else:
            slide = cv2.imread(slide_path)
            if slide is None:
                print('Could not read calibration slide: {}'.format(slide_path),
                      file=sys.stderr)
                return 1
            if args.vertical_core and not already_horizontal:
                slide = cv2.rotate(slide, cv2.ROTATE_90_COUNTERCLOCKWISE)
            composite = prepend_calibration_slide(composite, slide)
            print('Prepended calibration slide from {}'.format(slide_path))

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, composite)
    print('Wrote stitched image ({}x{}) to {}'.format(
        composite.shape[1], composite.shape[0], output_path))

    if args.preview:
        h, w = composite.shape[:2]
        if h > 1000 or w > 1000:
            scale = 1000 / max(h, w)
            preview = cv2.resize(composite, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        else:
            preview = composite
        cv2.imwrite(args.preview, preview)
        print('Wrote preview to {}'.format(args.preview))

    return 0


if __name__ == '__main__':
    sys.exit(main())