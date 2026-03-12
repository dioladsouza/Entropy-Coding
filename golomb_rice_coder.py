"""
Golomb-Rice Entropy Coder/Decoder
==================================
Encodes image pixel data using Golomb-Rice coding.

Pipeline:
  1. Load image as grayscale pixel array
  2. Compute residuals (delta encoding)
  3. Map residuals to non-negative integers (zigzag mapping)
  4. Choose optimal Rice parameter k from data statistics
  5. Encode each value: unary(quotient) + k-bit binary(remainder)
  6. Decode and reconstruct the original image
  7. Evaluate: compression ratio, encode/decode time, memory
"""

import time
import tracemalloc
import math
import struct
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
#   k = max(0, floor(log2(log2(mean + 1))))
# ─────────────────────────────────────────────

def optimal_k(values: list[int]) -> int:
    """Estimate the best Rice parameter k from the data mean."""
    mean = sum(values) / len(values) if values else 1
    if mean <= 1:
        return 0
    inner = math.log2(mean + 1)
    if inner <= 1:
        return 0
    return max(0, int(math.floor(math.log2(inner))))


# ─────────────────────────────────────────────
# STEP 3: Golomb-Rice encode a single integer
#   Given n and k:
#     q = n >> k          (floor division by 2^k)
#     r = n & (2^k - 1)   (remainder, lower k bits)
#   Output: unary(q) followed by k-bit binary(r)
# ─────────────────────────────────────────────

def rice_encode_value(n: int, k: int) -> str:
    """Encode a non-negative integer n with Rice parameter k.
    Returns a bit-string like '11010'.
    """
    q = n >> k              # quotient
    r = n & ((1 << k) - 1) # remainder (lower k bits)

    unary = '1' * q + '0'                         # q ones then a zero
    binary = format(r, f'0{k}b') if k > 0 else '' # k-bit binary remainder
    return unary + binary


def rice_decode_value(bits: str, pos: int, k: int) -> tuple[int, int]:
    """Decode one Rice-coded value starting at position pos in the bitstring.
    Returns (decoded_value, new_position).
    """
    # Read unary: count 1s until we hit a 0
    q = 0
    while pos < len(bits) and bits[pos] == '1':
        q += 1
        pos += 1
    pos += 1  # skip the terminating '0'

    # Read k-bit remainder
    r = int(bits[pos:pos + k], 2) if k > 0 else 0
    pos += k

    return (q << k) | r, pos


# ─────────────────────────────────────────────
# STEP 4: Encode an entire image
# ─────────────────────────────────────────────

def encode_image(image_path: str) -> dict:
    """
    Full Golomb-Rice encoding pipeline for a grayscale image.
    Returns a dict with bitstream, k, shape, and stats.
    """
    # Load image
    img = Image.open(image_path).convert('L')  # grayscale
    pixels = list(img.tobytes())
    height, width = img.size[1], img.size[0]

    # Delta (residual) encoding: predict each pixel from the previous one.
    # Cast to int to allow signed differences before zigzag mapping.
    pixels = [int(p) for p in pixels]
    residuals = [pixels[0]]  # first pixel stored as-is
    for i in range(1, len(pixels)):
        residuals.append(pixels[i] - pixels[i - 1])

    # Zigzag-map residuals to non-negative integers
    mapped = [zigzag_encode(r) for r in residuals]

    # Choose Rice parameter k
    k = optimal_k(mapped)

    # Encode each value
    bitstream = ''.join(rice_encode_value(v, k) for v in mapped)

    original_bits = len(pixels) * 8  # 8 bits per pixel (uint8)
    compressed_bits = len(bitstream)

    return {
        'bitstream': bitstream,
        'k': k,
        'width': width,
        'height': height,
        'original_bits': original_bits,
        'compressed_bits': compressed_bits,
        'compression_ratio': original_bits / compressed_bits,
        'num_pixels': len(pixels),
    }


# ─────────────────────────────────────────────
# STEP 5: Decode back to pixels
# ─────────────────────────────────────────────

def decode_image(result: dict) -> list[int]:
    """
    Decode a Golomb-Rice encoded bitstream back to pixel values.
    Returns a flat list of uint8 pixel values.
    """
    bitstream = result['bitstream']
    k = result['k']
    num_pixels = result['num_pixels']

    pos = 0
    mapped = []
    for _ in range(num_pixels):
        val, pos = rice_decode_value(bitstream, pos, k)
        mapped.append(val)

    # Reverse zigzag mapping
    residuals = [zigzag_decode(v) for v in mapped]

    # Reverse delta encoding
    pixels = [residuals[0]]
    for i in range(1, len(residuals)):
        pixels.append(pixels[-1] + residuals[i])

    # Clamp to valid uint8 range
    pixels = [max(0, min(255, p)) for p in pixels]
    return pixels


# ─────────────────────────────────────────────
# STEP 6: Evaluate and print results
# ─────────────────────────────────────────────

def evaluate(image_path: str):
    print(f"\n{'='*52}")
    print(f"  Golomb-Rice Coder — {image_path}")
    print(f"{'='*52}")

    # ── Encoding ──
    tracemalloc.start()
    t0 = time.perf_counter()
    result = encode_image(image_path)
    encode_time = time.perf_counter() - t0
    _, encode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Decoding ──
    tracemalloc.start()
    t1 = time.perf_counter()
    decoded_pixels = decode_image(result)
    decode_time = time.perf_counter() - t1
    _, decode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Verify lossless reconstruction ──
    img = Image.open(image_path).convert('L')
    original_pixels = list(img.tobytes())
    lossless = (decoded_pixels == original_pixels)

    # ── Save reconstructed image ──
    out_img = Image.new('L', (result['width'], result['height']))
    out_img.putdata(decoded_pixels)
    out_path = image_path.replace('.', '_decoded.')
    out_img.save(out_path)

    # ── Print report ──
    print(f"  Image size        : {result['width']} x {result['height']} px")
    print(f"  Rice parameter k  : {result['k']}")
    print(f"  Original size     : {result['original_bits'] // 8:,} bytes")
    print(f"  Compressed size   : {result['compressed_bits'] // 8:,} bytes")
    print(f"  Compression ratio : {result['compression_ratio']:.3f}x")
    print(f"  Space saving      : {(1 - 1/result['compression_ratio'])*100:.1f}%")
    print(f"  Encode time       : {encode_time*1000:.2f} ms")
    print(f"  Decode time       : {decode_time*1000:.2f} ms")
    print(f"  Encode memory     : {encode_peak / 1024:.1f} KB")
    print(f"  Decode memory     : {decode_peak / 1024:.1f} KB")
    print(f"  Lossless check    : {'PASS ✓' if lossless else 'FAIL ✗'}")
    print(f"  Decoded saved to  : {out_path}")
    print(f"{'='*52}\n")

    return result


# ─────────────────────────────────────────────
# STEP 7: Generate a synthetic test image
#   (in case you don't have a test image handy)
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
        # Use provided image path
        image_path = sys.argv[1]
    else:
        # Generate a synthetic test image
        image_path = '/tmp/test_image.png'
        generate_test_image(image_path)

    if not os.path.exists(image_path):
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    evaluate(image_path)