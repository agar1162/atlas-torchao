import torch

def hw_bf16_to_e4m3(x: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Bit-exact vectorized BF16ScaleToE4M3. scale_e8m0 must broadcast against x."""
    b = x.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign, e, f = (b >> 15) & 1, (b >> 7) & 0xFF, b & 0x7F

    mant8 = 0x80 | f
    trunc = mant8 >> 4
    inc = ((mant8 >> 3) & 1) & (((mant8 & 7) != 0) | (trunc & 1)).int()   # RNE
    rounded = trunc + inc
    carry = rounded == 16

    final_exp = (e - 127) + (scale_e8m0.to(torch.int32) - 127) + carry.int()
    mant = torch.where(carry, 0, (rounded - 8) & 7)

    normal = (sign << 7) | (((final_exp + 7) & 0xF) << 3) | mant
    sat = (sign << 7) | 0x7E
    out = torch.where(final_exp > 8, sat, torch.where(final_exp >= -6, normal, 0))
    out = torch.where((e == 0xFF) & (f == 0), sat, out)               # ±Inf saturates
    out = torch.where((e == 0) | ((e == 0xFF) & (f != 0)), 0, out)    # zero/subnormal/NaN -> 0
    return out.to(torch.uint8)


def hw_e4m3_to_float(q: torch.Tensor) -> torch.Tensor:
    """Bit-exact e4m3ToBf16 (subnormals and 0x7F/0xFF flush to signed zero)."""
    q = q.to(torch.int32)
    exp, mant = (q >> 3) & 0xF, q & 7
    val = (1 + mant / 8.0) * torch.exp2((exp - 7).float())
    val = torch.where((q >> 7) & 1 == 1, -val, val)
    flush = (exp == 0) | ((exp == 0xF) & (mant == 7))
    return torch.where(flush, torch.copysign(torch.zeros_like(val), val), val)


def scale_exp_for(t: torch.Tensor) -> torch.Tensor:
    """Pick a power-of-two scale so the largest value fits in E4M3 (max 448)."""
    amax = t.detach().abs().max().float().clamp_min(2.0**-126)
    return torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127).to(torch.int32)
