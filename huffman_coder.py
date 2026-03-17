"""
Huffman Entropy Coder/Decoder
==============================
Encodes image pixel data using Huffman coding.

Pipeline:
  1. Load image as grayscale pixel array
  2. Compute 2D residuals using MED predictor (JPEG-LS style)
  3. Map residuals to non-negative integers (zigzag mapping)
  4. Build Huffman tree based on frequency of mapped residuals
  5. Encode into bitstream
  6. Decode and reconstruct original image
  7. Evaluate: CR, time, memory
"""

import time
import tracemalloc
import heapq
from collections import Counter
from PIL import Image
import os

# ─────────────────────────────────────────────
# STEP 1: Helpers (Zigzag & MED Predictor)
# 保持和 Golomb/CABAC 一致，确保对比公平
# ─────────────────────────────────────────────

def zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 31)

def zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)

def med_predict(a, b, c):
    if c >= max(a, b): return min(a, b)
    elif c <= min(a, b): return max(a, b)
    else: return a + b - c

def compute_residuals_2d(pixels, width, height):
    residuals = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x == 0 and y == 0: pred = 128
            elif y == 0: pred = pixels[idx - 1]
            elif x == 0: pred = pixels[idx - width]
            else:
                a, b, c = pixels[idx-1], pixels[idx-width], pixels[idx-width-1]
                pred = med_predict(a, b, c)
            residuals.append(pixels[idx] - pred)
    return residuals

def reconstruct_pixels_2d(residuals, width, height):
    pixels = [0] * (width * height)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x == 0 and y == 0: pred = 128
            elif y == 0: pred = pixels[idx - 1]
            elif x == 0: pred = pixels[idx - width]
            else:
                a, b, c = pixels[idx-1], pixels[idx-width], pixels[idx-width-1]
                pred = med_predict(a, b, c)
            pixels[idx] = residuals[idx] + pred
    return pixels

# ─────────────────────────────────────────────
# STEP 2: Huffman Core Logic
# ─────────────────────────────────────────────

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freqs = Counter(data)
    heap = [HuffmanNode(char, freq) for char, freq in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0] if heap else None

def build_codes(node, prefix="", code_map={}):
    if node is not None:
        if node.char is not None:
            code_map[node.char] = prefix
        build_codes(node.left, prefix + "0", code_map)
        build_codes(node.right, prefix + "1", code_map)
    return code_map

# ─────────────────────────────────────────────
# STEP 3: Encode / Decode Interface
# ─────────────────────────────────────────────

def encode_image(image_path: str):
    img = Image.open(image_path).convert('L')
    pixels = [int(p) for p in img.tobytes()]
    width, height = img.size

    residuals = compute_residuals_2d(pixels, width, height)
    mapped = [zigzag_encode(r) for r in residuals]

    # Build Huffman Tree & Codes
    root = build_huffman_tree(mapped)
    codes = build_codes(root)
    
    # Generate bitstream
    bitstream = "".join(codes[v] for v in mapped)
    
    return {
        'bitstream': bitstream,
        'huffman_root': root, # Decoder needs this
        'width': width,
        'height': height,
        'original_bits': len(pixels) * 8,
        'compressed_bits': len(bitstream),
        'compression_ratio': (len(pixels) * 8) / len(bitstream) if len(bitstream) > 0 else 0,
        'num_pixels': len(pixels)
    }

def decode_image(result: dict):
    bitstream = result['bitstream']
    root = result['huffman_root']
    
    # Decode bitstream using tree
    mapped = []
    curr = root
    for bit in bitstream:
        curr = curr.left if bit == '0' else curr.right
        if curr.char is not None:
            mapped.append(curr.char)
            curr = root
            
    residuals = [zigzag_decode(v) for v in mapped]
    pixels = reconstruct_pixels_2d(residuals, result['width'], result['height'])
    return [max(0, min(255, p)) for p in pixels]

# ─────────────────────────────────────────────
# STEP 4: Standard Evaluate Interface (Matches Benchmark)
# ─────────────────────────────────────────────

def evaluate(image_path: str):
    print(f"\n{'='*52}")
    print(f"  Huffman Coder — {image_path}")
    print(f"{'='*52}")

    tracemalloc.start()
    t0 = time.perf_counter()
    result = encode_image(image_path)
    encode_time = time.perf_counter() - t0
    _, encode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t1 = time.perf_counter()
    decoded_pixels = decode_image(result)
    decode_time = time.perf_counter() - t1
    _, decode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Save outputs
    out_dir = "result_images"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    base_name = os.path.basename(image_path)
    out_path = os.path.join(out_dir, base_name.replace('.png', '_huffman_decoded.png'))
    
    out_img = Image.new('L', (result['width'], result['height']))
    out_img.putdata(decoded_pixels)
    out_img.save(out_path)

    # Lossless Check
    orig = list(Image.open(image_path).convert('L').tobytes())
    lossless = (decoded_pixels == orig)

    cr = result['compression_ratio']
    print(f"  Image size        : {result['width']} x {result['height']} px")
    print(f"  Original size     : {result['original_bits']//8:,} bytes")
    print(f"  Compressed size   : {result['compressed_bits']//8:,} bytes")
    print(f"  Compression ratio : {cr:.3f}x")
    print(f"  Space saving      : {(1-1/cr)*100:.1f}%" if cr > 1 else "  Space saving      : n/a")
    print(f"  Encode time       : {encode_time*1000:.2f} ms")
    print(f"  Decode time       : {decode_time*1000:.2f} ms")
    print(f"  Encode memory     : {encode_peak/1024:.1f} KB")
    print(f"  Decode memory     : {decode_peak/1024:.1f} KB")
    print(f"  Lossless check    : {'PASS ✓' if lossless else 'FAIL ✗'}")
    print(f"  Decoded saved to  : {out_path}")
    print(f"{'='*52}\n")

    return {**result, 'encode_time': encode_time, 'decode_time': decode_time,
            'encode_peak_kb': encode_peak/1024, 'decode_peak_kb': decode_peak/1024,
            'lossless': lossless}

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1: evaluate(sys.argv[1])