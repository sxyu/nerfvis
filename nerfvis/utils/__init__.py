from typing import Dict

import numpy as np

try:
    from ._rotation import Rotation
except:
    from scipy.spatial.transform import Rotation # If cython not available, requires scipy

def _expand_bits(v):
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;

def _unexpand_bits(v):
    v &= 0x49249249;
    v = (v | (v >> 2)) & 0xc30c30c3;
    v = (v | (v >> 4)) & 0xf00f00f;
    v = (v | (v >> 8)) & 0xff0000ff;
    v = (v | (v >> 16)) & 0x0000ffff;
    return v;

def morton(x, y, z):
    xx = _expand_bits(x);
    yy = _expand_bits(y);
    zz = _expand_bits(z);
    return (xx << 2) + (yy << 1) + zz;

def inv_morton(code):
    x = _unexpand_bits(code >> 2);
    y = _unexpand_bits(code >> 1);
    z = _unexpand_bits(code);
    return x, y, z

def morton_grid(pow2) -> np.ndarray:
    mg = np.mgrid[:pow2, :pow2, :pow2].reshape(3, -1)
    return morton(*mg)

def inv_morton_grid(pow2 : int) -> np.ndarray:
    mg = np.arange(pow2 ** 3)
    x, y, z = inv_morton(mg)
    return (x * pow2 + y) * pow2 + z

def vol2plenoctree(
            density : np.ndarray,
            colors : np.ndarray,
            radius: float = 1.0,
            density_threshold: float = 1.0,
            data_format : str = "RGBA") -> Dict[str, np.ndarray]:
    """
    Convert arbirary volume to PlenOctree

    :param density: (Dx, Dy, Dz); dimensions need not be powers
                                  of 2 nor equal
    :param colors: (Dx, Dy, Dz, (3, channel_size));
                                  color data,
                                  last dim is size 3 * channel_size
    :param radius: 1/2 side length of volume
    :param density_threshold: threshold below which
                              density is ignored
    :param data_format: standard PlenOctree data format string,
                        one of :code:`RGBA | SH1 | SH4 | SH9 | SH16`.
                        The channel_size should be respectively
                        :code:`1 | 1 | 4 | 9 | 16```.

    :return: Dict, generated PlenOctree data with all keys required
                   by the renderer. You can save this with
                   :code:`np.savez_compressed("name.npz", **returned_data)`
                   or
                   :code:`np.savez("name.npz", **returned_data)`
                   and open it with volrend.
    """
    # Check dimensions
    assert density.ndim == 3
    assert colors.ndim == 4 and tuple(colors.shape[:3]) == tuple(density.shape), \
            f"{density.shape} != {colors.shape[:3]}"

    dims = list(density.shape)
    maxdim = max(dims)
    assert maxdim <= 1024, "Voxel grid too large"
    n_levels = (maxdim - 1).bit_length()

    # Check data formats
    valid_data_formats = {
        "RGBA" : 4,
        "SH1" : 4,
        "SH4" : 13,
        "SH9" : 28,
        "SH16" : 49,
    }
    assert data_format in valid_data_formats, f"Invalid ddata format {data_format}"
    data_dim = valid_data_formats[data_format]

    # Check if given shape matches promised format
    assert colors.shape[-1] + 1 == data_dim

    result = {}
    result['data_dim'] = np.int64(data_dim)
    result['data_format'] = data_format
    result['invradius3'] = np.array([
             0.5 / radius,
             0.5 / radius,
             0.5 / radius], dtype=np.float32)

    require_pad = dims != [2 ** n_levels] * 3
    center = np.array([radius * (1.0 - dim / 2 ** n_levels) for dim in dims],
                      dtype=np.float32)

    result['offset'] = (0.5 * (1.0 - center / radius)).astype(
            np.float32)

    # Construct mask hierarchy
    hierarchy = []
    pow2 = 2 ** n_levels
    mask = np.zeros((pow2, pow2, pow2), dtype=bool)
    density_mask = density > density_threshold
    mask[:dims[0], :dims[1], :dims[2]] = density_mask

    hierarchy.append(mask)
    while pow2 > 1:
        mask = mask.reshape((pow2 // 2, 2, pow2 // 2, 2, pow2 // 2, 2))
        mask = mask.any(axis=(1, 3, 5))
        pow2 //= 2
        hierarchy.append(mask)

    hierarchy = hierarchy[::-1]

    # PlenOctree standard format data arrays
    all_child = []
    all_data = []
    mg, img = morton_grid(1), inv_morton_grid(1)
    curr_indices = np.zeros(1, dtype=np.uint32)
    for i, (mask, next_mask) in enumerate(zip(hierarchy[:-1], hierarchy[1:])):
        nnodes = mask.sum()
        pow2 = mask.shape[0]

        if i == len(hierarchy) - 2:
            # Construct the last tree level
            child = np.zeros((nnodes, 2, 2, 2), dtype=np.uint32);
            if require_pad:
                # Data is not power of 2, pad it
                npow2 = 2 * pow2
                density = np.pad(density,
                                 [(0, npow2 - dims[0]), (0, npow2 - dims[1]),
                                  (0, npow2 - dims[2])])
                colors = np.pad(colors,
                                [(0, npow2 - dims[0]), (0, npow2 - dims[1]),
                                 (0, npow2 - dims[2]), (0, 0)])

            mask_indices = curr_indices[mask.reshape(-1)]
            density_i = np.empty((nnodes, 2, 2, 2, 1), dtype=np.float16);
            density_i[mask_indices] = density.reshape(
                    pow2, 2, pow2, 2, pow2, 2, 1).transpose(
                            0, 2, 4, 1, 3, 5, 6).reshape(
                            -1, 2, 2, 2, 1)[mask.flatten()].astype(np.float16)
            colors_i = np.empty((nnodes, 2, 2, 2, data_dim - 1), dtype=np.float16);
            colors_i[mask_indices] = colors.reshape(
                    pow2, 2, pow2, 2, pow2, 2, data_dim - 1).transpose(
                            0, 2, 4, 1, 3, 5, 6).reshape(
                            -1, 2, 2, 2, data_dim - 1)[mask.flatten()].astype(np.float16)
            data = np.concatenate([colors_i.astype(np.float16),
                                   density_i.astype(np.float16)], -1)
        else:
            # Construct an internal level
            mg, img = morton_grid(pow2 * 2), inv_morton_grid(pow2 * 2)
            next_indices = np.cumsum(next_mask.reshape(-1)[img], dtype=np.uint32)[mg] - 1
            next_indices[~next_mask.reshape(-1)] = 0

            child = (next_indices.reshape(pow2, 2, pow2, 2, pow2, 2) + nnodes -
                     curr_indices.reshape(pow2, 1, pow2, 1, pow2, 1))
            child = child.reshape(pow2 * 2, pow2 * 2, pow2 * 2)
            child[~next_mask] = 0
            child = child.reshape(pow2, 2, pow2, 2, pow2, 2)
            child = child.transpose(0, 2, 4, 1, 3, 5).reshape(pow2 ** 3, 2, 2, 2)

            child_tmp = np.empty((nnodes, 2, 2, 2), dtype=np.uint32);
            child_tmp[curr_indices[mask.reshape(-1)]] = child[mask.reshape(-1)]
            child = child_tmp

            # For now, all interior nodes will be empty
            data = np.zeros((nnodes, 2, 2, 2, data_dim), dtype=np.float16);
            curr_indices = next_indices

        all_child.append(child)
        all_data.append(data)

    child = np.concatenate(all_child, 0)
    data = np.concatenate(all_data, 0)
    result['child'] = child.view(np.int32)
    result['data'] = data
    return result


__all__ = ('Rotation', 'vol2plenoctree')
