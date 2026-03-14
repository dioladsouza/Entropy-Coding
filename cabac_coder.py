"""
CABAC (Context-Adaptive Binary Arithmetic Coding) Implementation
=================================================================
A faithful simplified CABAC encoder/decoder inspired by H.264/H.265.

Key components:
  - Binary arithmetic encoder/decoder (integer, 32-bit precision)
  - 64-state adaptive context model (LPS probability state machine)
  - Unary binarization with fixed-length suffix for large values
  - Context selection per bin position
  - Delta encoding + zigzag mapping (same as Golomb-Rice)
"""

import time, tracemalloc, math
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# Arithmetic Coder – correct integer implementation
# Encodes symbols into a bit sequence using interval subdivision.
# ─────────────────────────────────────────────

PRECISION = 32
HALF      = 1 << (PRECISION - 1)   # 0x80000000
QUARTER   = 1 << (PRECISION - 2)   # 0x40000000
FULL      = (1 << PRECISION) - 1   # 0xFFFFFFFF

class ArithEncoder:
    def __init__(self):
        self.low    = 0
        self.high   = FULL
        self.bits   = []
        self.pending = 0

    def encode(self, symbol: int, prob0: float):
        """Encode one bit. prob0 = probability that symbol is 0."""
        rng = self.high - self.low + 1
        split = self.low + max(1, int(rng * prob0)) - 1

        if symbol == 0:
            self.high = split
        else:
            self.low = split + 1

        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low  -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.pending += 1
                self.low  -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low  <<= 1
            self.high  = (self.high << 1) | 1

    def _emit(self, b: int):
        self.bits.append(b)
        while self.pending:
            self.bits.append(1 - b)
            self.pending -= 1

    def flush(self):
        self.pending += 1
        if self.low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self.bits


class ArithDecoder:
    def __init__(self, bits: list):
        self.bits  = bits
        self.pos   = 0
        self.low   = 0
        self.high  = FULL
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self._read()

    def _read(self):
        if self.pos < len(self.bits):
            b = self.bits[self.pos]; self.pos += 1; return b
        return 0

    def decode(self, prob0: float) -> int:
        rng   = self.high - self.low + 1
        split = self.low + max(1, int(rng * prob0)) - 1
        symbol = 0 if self.value <= split else 1

        if symbol == 0:
            self.high = split
        else:
            self.low = split + 1

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low   -= HALF
                self.high  -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.low   -= QUARTER
                self.high  -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low   <<= 1
            self.high   = (self.high << 1) | 1
            self.value  = (self.value << 1) | self._read()

        return symbol


# ─────────────────────────────────────────────
# CABAC context model: 64-state LPS probability table
# Each state tracks MPS (most probable symbol) and transitions
# ─────────────────────────────────────────────

# LPS probability for each of 64 states (from H.264 spec, simplified)
LPS_PROB = [
    0.500, 0.469, 0.438, 0.406, 0.375, 0.344, 0.313, 0.288,
    0.263, 0.238, 0.219, 0.200, 0.188, 0.169, 0.156, 0.144,
    0.131, 0.119, 0.109, 0.100, 0.091, 0.081, 0.075, 0.069,
    0.063, 0.056, 0.050, 0.047, 0.044, 0.041, 0.038, 0.034,
    0.031, 0.028, 0.025, 0.023, 0.022, 0.020, 0.019, 0.017,
    0.016, 0.014, 0.013, 0.012, 0.011, 0.010, 0.009, 0.009,
    0.008, 0.007, 0.006, 0.006, 0.006, 0.005, 0.005, 0.004,
    0.004, 0.003, 0.003, 0.003, 0.002, 0.002, 0.001, 0.001,
]

class ContextModel:
    __slots__ = ('state', 'mps')

    def __init__(self):
        self.state = 0  # maximum uncertainty
        self.mps   = 0  # MPS = 0 initially

    def prob0(self) -> float:
        """Return P(0) for the arithmetic coder."""
        if self.mps == 0:
            return 1.0 - LPS_PROB[self.state]   # P(MPS=0) = 1 - P(LPS)
        else:
            return LPS_PROB[self.state]          # P(MPS=1) → P(0) = P(LPS)

    def update(self, bit: int):
        """Update state after observing bit."""
        if bit == self.mps:
            # MPS observed → increase confidence (higher state)
            self.state = min(self.state + 1, 63)
        else:
            # LPS observed → decrease confidence (lower state)
            if self.state <= 1:
                self.mps ^= 1   # flip MPS when nearly uncertain
            self.state = max(self.state - 2, 0)


# ─────────────────────────────────────────────
# Binarization: unary prefix, fixed-length suffix
# ─────────────────────────────────────────────

MAX_UNARY    = 16   # unary prefix threshold
SUFFIX_BITS  = 9    # 9-bit suffix → can encode values 0..16+511=527 (covers zigzag range 0..510)

def binarize(value: int) -> list:
    if value < MAX_UNARY:
        return [1] * value + [0]
    prefix = [1] * MAX_UNARY + [0]
    suffix = [((value - MAX_UNARY) >> (SUFFIX_BITS - 1 - i)) & 1 for i in range(SUFFIX_BITS)]
    return prefix + suffix

def debinarize_from_decoder(dec: ArithDecoder, contexts: list) -> int:
    """Read bins one at a time and recover value."""
    count = 0
    for i in range(MAX_UNARY):
        ctx = contexts[i]
        bit = dec.decode(ctx.prob0())
        ctx.update(bit)
        if bit == 0:
            return count
        count += 1
    # count == MAX_UNARY; read terminating 0
    ctx = contexts[MAX_UNARY]
    bit = dec.decode(ctx.prob0())
    ctx.update(bit)
    # Read SUFFIX_BITS-bit fixed suffix
    val = 0
    for i in range(SUFFIX_BITS):
        ctx = contexts[MAX_UNARY + 1 + i]
        bit = dec.decode(ctx.prob0())
        ctx.update(bit)
        val = (val << 1) | bit
    return MAX_UNARY + val


NUM_CONTEXTS = MAX_UNARY + 1 + SUFFIX_BITS  # 26 context models


# ─────────────────────────────────────────────
# Zigzag helpers
# ─────────────────────────────────────────────

def zigzag_encode(v: int) -> int:
    return (v << 1) ^ (v >> 31)

def zigzag_decode(v: int) -> int:
    return (v >> 1) ^ -(v & 1)


# ─────────────────────────────────────────────
# Encode pipeline
# ─────────────────────────────────────────────

def encode_image(image_path: str) -> dict:
    img    = Image.open(image_path).convert('L')
    pixels = [int(p) for p in img.tobytes()]
    width, height = img.size

    residuals = [pixels[0]] + [pixels[i] - pixels[i-1] for i in range(1, len(pixels))]
    mapped    = [zigzag_encode(r) for r in residuals]

    contexts = [ContextModel() for _ in range(NUM_CONTEXTS)]
    enc      = ArithEncoder()

    for value in mapped:
        bins = binarize(value)
        for bin_idx, bit in enumerate(bins):
            ctx = contexts[min(bin_idx, NUM_CONTEXTS - 1)]
            enc.encode(bit, ctx.prob0())
            ctx.update(bit)

    bitstream = enc.flush()

    orig_bits = len(pixels) * 8
    comp_bits = len(bitstream)

    return {
        'bitstream'       : bitstream,
        'width'           : width,
        'height'          : height,
        'original_bits'   : orig_bits,
        'compressed_bits' : comp_bits,
        'compression_ratio': orig_bits / comp_bits if comp_bits > 0 else 0,
        'num_pixels'      : len(pixels),
    }


# ─────────────────────────────────────────────
# Decode pipeline
# ─────────────────────────────────────────────

def decode_image(result: dict) -> list:
    contexts = [ContextModel() for _ in range(NUM_CONTEXTS)]
    dec      = ArithDecoder(result['bitstream'])

    mapped = [debinarize_from_decoder(dec, contexts) for _ in range(result['num_pixels'])]

    residuals = [zigzag_decode(v) for v in mapped]
    pixels    = [residuals[0]]
    for i in range(1, len(residuals)):
        pixels.append(pixels[-1] + residuals[i])
    return [max(0, min(255, p)) for p in pixels]


# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────

def evaluate(image_path: str) -> dict:
    print(f"\n{'='*52}")
    print(f"  CABAC Coder — {image_path}")
    print(f"{'='*52}")

    tracemalloc.start()
    t0     = time.perf_counter()
    result = encode_image(image_path)
    encode_time = time.perf_counter() - t0
    _, encode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t1      = time.perf_counter()
    decoded = decode_image(result)
    decode_time = time.perf_counter() - t1
    _, decode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    orig    = list(Image.open(image_path).convert('L').tobytes())
    lossless = (decoded == orig)

    out_img = Image.new('L', (result['width'], result['height']))
    out_img.putdata(decoded)
    out_path = image_path.replace('.png', '_cabac_decoded.png')
    out_img.save(out_path)

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
    print(f"{'='*52}\n")

    return {**result, 'encode_time': encode_time, 'decode_time': decode_time,
            'encode_peak_kb': encode_peak/1024, 'decode_peak_kb': decode_peak/1024,
            'lossless': lossless}


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/test_gradient.png'
    evaluate(path)