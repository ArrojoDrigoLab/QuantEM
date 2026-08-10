"""Dependency-free readers used by dataset drivers (no nibabel/SimpleITK)."""
import gzip, struct
import numpy as np

_NII_DTYPE = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
              256: np.int8, 512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}


def read_nii_gz(path):
    """Minimal NIfTI-1 .nii.gz reader. Returns (array[X,Y,Z...], zooms, (slope,inter))."""
    with gzip.open(path, "rb") as f:
        buf = f.read()
    sizeof_hdr = struct.unpack_from("<i", buf, 0)[0]
    e = "<" if sizeof_hdr == 348 else ">"
    dim = struct.unpack_from(e + "8h", buf, 40)
    datatype = struct.unpack_from(e + "h", buf, 70)[0]
    pixdim = struct.unpack_from(e + "8f", buf, 76)
    vox_offset = int(struct.unpack_from(e + "f", buf, 108)[0])
    slope, inter = struct.unpack_from(e + "2f", buf, 112)
    ndim = dim[0]
    shape = [dim[i] for i in range(1, ndim + 1)]
    dt = np.dtype(_NII_DTYPE[datatype]).newbyteorder(e)
    n = int(np.prod(shape))
    data = np.frombuffer(buf, dtype=dt, count=n, offset=vox_offset)
    arr = data.reshape(shape, order="F").astype(_NII_DTYPE[datatype])
    zooms = [pixdim[i] for i in range(1, ndim + 1)]
    return arr, zooms, (slope, inter)


def to_uint8_fullrange(a):
    """Min/max scale to uint8 (canonical full-range 8-bit, no windowing)."""
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
