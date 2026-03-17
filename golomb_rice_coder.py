"""
Golomb-Rice Entropy Coder/Decoder
==================================
Encodes image pixel data using Golomb-Rice coding.

Pipeline:
  1. Load image as grayscale pixel array
  2. Compute 2D residuals using MED predictor (JPEG-LS style)
  3. Map residuals to non-negative integers (zigzag mapping)
  4. Choose optimal Rice parameter k per row from data statistics
  5. Encode each value: unary(quotient) + k-bit binary(remainder)
  6. Decode and reconstruct the original image
  7. Evaluate: compression ratio, encode/decode time, memory
"""

import time
import tracemalloc
import math
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# STEP 1: Zigzag mapping  (signed → non-negative)
#   Maps: 0→0, -1→1, 1→2, -2→3, 2→4, ...
# ─────────────────────────────────────────────

def zigzag_encode(value: int) -> int:
    """Map a signed integer to a non-negative integer."""
    return (value << 1) ^ (value >> 31)

def zigzag_decode(value: int) -> int:
    """Reverse the zigzag mapping back to a signed integer."""
    return (value >> 1) ^ -(value & 1)


# ─────────────────────────────────────────────
# STEP 2: Choose optimal Rice parameter k
#   Standard formula used in FLAC:
#   k = round(log2(mean)), clamped to [0, 8]
#   Much more accurate than the log2(log2()) heuristic
#   for natural images with larger residuals.
# ─────────────────────────────────────────────

def optimal_k(values: list) -> int:
    """Estimate the best Rice parameter k from the data mean."""
    if not values:
        return 0
    mean = sum(values) / len(values)
    if mean <= 1:
        return 0
    return max(0, min(8, round(math.log2(mean))))


# ─────────────────────────────────────────────
# STEP 3: 2D MED predictor (LOCO-I / JPEG-LS)
#   Uses left, above, and above-left neighbours
#   to predict each pixel. Much better than 1D
#   left-only delta for natural images because it
#   exploits both horizontal and vertical correlation.
#
#   MED rule:
#     if above-left >= max(left, above): predict = min(left, above)
#     if above-left <= min(left, above): predict = max(left, above)
#     else:                              predict = left + above - above-left
# ─────────────────────────────────────────────

def med_predict(a: int, b: int, c: int) -> int:
    """LOCO-I MED predictor.
    a = left, b = above, c = above-left
    """
    if c >= max(a, b):
        return min(a, b)
    elif c <= min(a, b):
        return max(a, b)
    else:
        return a + b - c


def compute_residuals_2d(pixels: list, width: int, height: int) -> list:
    """
    Compute prediction residuals using the 2D MED predictor.
    Scans left-to-right, top-to-bottom.

    Edge cases:
      - Top-left pixel: predict 128 (mid-grey DC offset)
      - Top row (y=0, x>0): predict from left neighbour only
      - Left column (x=0, y>0): predict from above neighbour only
      - All other pixels: use full MED predictor
    """
    residuals = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x == 0 and y == 0:
                pred = 128
            elif y == 0:
                pred = pixels[idx - 1]           # top row: left only
            elif x == 0:
                pred = pixels[idx - width]       # left col: above only
            else:
                a = pixels[idx - 1]              # left
                b = pixels[idx - width]          # above
                c = pixels[idx - width - 1]      # above-left
                pred = med_predict(a, b, c)
            residuals.append(pixels[idx] - pred)
    return residuals


def reconstruct_pixels_2d(residuals: list, width: int, height: int) -> list:
    """
    Reverse the 2D MED prediction — must mirror compute_residuals_2d exactly.
    Reconstructs pixels in the same scan order using already-decoded neighbours.
    """
    pixels = [0] * (width * height)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x == 0 and y == 0:
                pred = 128
            elif y == 0:
                pred = pixels[idx - 1]
            elif x == 0:
                pred = pixels[idx - width]
            else:
                a = pixels[idx - 1]
                b = pixels[idx - width]
                c = pixels[idx - width - 1]
                pred = med_predict(a, b, c)
            pixels[idx] = residuals[idx] + pred
    return pixels


# ─────────────────────────────────────────────
# STEP 4: Golomb-Rice encode / decode one value
# ─────────────────────────────────────────────

def rice_encode_value(n: int, k: int) -> str:
    """Encode a non-negative integer n with Rice parameter k.
    Returns a bit-string like '11010'.
    """
    q = n >> k               # quotient
    r = n & ((1 << k) - 1)  # remainder (lower k bits)
    unary  = '1' * q + '0'
    binary = format(r, f'0{k}b') if k > 0 else ''
    return unary + binary


def rice_decode_value(bits: str, pos: int, k: int) -> tuple:
    """Decode one Rice-coded value starting at position pos.
    Returns (decoded_value, new_position).
    """
    q = 0
    while pos < len(bits) and bits[pos] == '1':
        q += 1
        pos += 1
    pos += 1  # skip terminating '0'
    r = int(bits[pos:pos + k], 2) if k > 0 else 0
    pos += k
    return (q << k) | r, pos


# ─────────────────────────────────────────────
# STEP 5: Encode an entire image
#   - 2D MED residuals
#   - Per-row adaptive k selection
#   - k_values stored alongside bitstream for decoder
# ─────────────────────────────────────────────

def encode_image(image_path: str) -> dict:
    """
    Full Golomb-Rice encoding pipeline for a grayscale image.
    Uses 2D MED predictor and per-row adaptive k selection.
    Returns a dict with bitstream, k_values, shape, and stats.
    """
    img    = Image.open(image_path).convert('L')
    pixels = [int(p) for p in img.tobytes()]
    width, height = img.size

    # 2D residuals + zigzag mapping
    residuals = compute_residuals_2d(pixels, width, height)
    mapped    = [zigzag_encode(r) for r in residuals]

    # Per-row k selection and encoding
    bitstream_parts = []
    k_values        = []

    for row in range(height):
        row_vals = mapped[row * width : (row + 1) * width]
        k        = optimal_k(row_vals)
        k_values.append(k)
        for v in row_vals:
            bitstream_parts.append(rice_encode_value(v, k))

    bitstream = ''.join(bitstream_parts)

    original_bits   = len(pixels) * 8
    compressed_bits = len(bitstream)

    return {
        'bitstream'        : bitstream,
        'k_values'         : k_values,   # one k per row — needed by decoder
        'k'                : k_values[-1] if k_values else 0,  # for display
        'width'            : width,
        'height'           : height,
        'original_bits'    : original_bits,
        'compressed_bits'  : compressed_bits,
        'compression_ratio': original_bits / compressed_bits if compressed_bits else 0,
        'num_pixels'       : len(pixels),
    }


# ─────────────────────────────────────────────
# STEP 6: Decode back to pixels
# ─────────────────────────────────────────────

def decode_image(result: dict) -> list:
    """
    Decode a Golomb-Rice encoded bitstream back to pixel values.
    Uses per-row k_values to mirror the adaptive encoder.
    Returns a flat list of uint8 pixel values.
    """
    bitstream  = result['bitstream']
    k_values   = result['k_values']
    width      = result['width']
    height     = result['height']

    pos    = 0
    mapped = []

    for row in range(height):
        k = k_values[row]
        for _ in range(width):
            val, pos = rice_decode_value(bitstream, pos, k)
            mapped.append(val)

    # Reverse zigzag
    residuals = [zigzag_decode(v) for v in mapped]

    # Reverse 2D MED prediction
    pixels = reconstruct_pixels_2d(residuals, width, height)

    return [max(0, min(255, p)) for p in pixels]


# ─────────────────────────────────────────────
# STEP 7: Evaluate and print results
# ─────────────────────────────────────────────

def evaluate(image_path: str):
    print(f"\n{'='*52}")
    print(f"  Golomb-Rice Coder — {image_path}")
    print(f"{'='*52}")

    tracemalloc.start()
    t0     = time.perf_counter()
    result = encode_image(image_path)
    encode_time = time.perf_counter() - t0
    _, encode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t1             = time.perf_counter()
    decoded_pixels = decode_image(result)
    decode_time    = time.perf_counter() - t1
    _, decode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    img             = Image.open(image_path).convert('L')
    original_pixels = list(img.tobytes())
    lossless        = (decoded_pixels == original_pixels)

    out_img = Image.new('L', (result['width'], result['height']))
    out_img.putdata(decoded_pixels)
    out_path = image_path.replace('.', '_decoded.')
    out_img.save(out_path)

    cr     = result['compression_ratio']
    k_vals = result['k_values']

    print(f"  Image size        : {result['width']} x {result['height']} px")
    print(f"  Rice parameter k  : {min(k_vals)}–{max(k_vals)} (per-row adaptive)")
    print(f"  Original size     : {result['original_bits'] // 8:,} bytes")
    print(f"  Compressed size   : {result['compressed_bits'] // 8:,} bytes")
    print(f"  Compression ratio : {cr:.3f}x")
    if cr > 1:
        print(f"  Space saving      : {(1 - 1/cr)*100:.1f}%")
    else:
        print(f"  Space saving      : {(1 - 1/cr)*100:.1f}% (expansion)")
    print(f"  Encode time       : {encode_time*1000:.2f} ms")
    print(f"  Decode time       : {decode_time*1000:.2f} ms")
    print(f"  Encode memory     : {encode_peak / 1024:.1f} KB")
    print(f"  Decode memory     : {decode_peak / 1024:.1f} KB")
    print(f"  Lossless check    : {'PASS ✓' if lossless else 'FAIL ✗'}")
    print(f"  Decoded saved to  : {out_path}")
    print(f"{'='*52}\n")

    return result


# ─────────────────────────────────────────────
# STEP 8: Generate synthetic test images
# ─────────────────────────────────────────────

def generate_test_image(path: str = 'test_image.png', size: int = 256):
    """Create a smooth gradient test image for benchmarking."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            arr[i, j] = int((i + j) / (2 * size) * 255)
    Image.fromarray(arr, mode='L').save(path)
    print(f"Test image saved to: {path}")
    return path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = '/tmp/test_image.png'
        generate_test_image(image_path)

    if not os.path.exists(image_path):
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    evaluate(image_path)