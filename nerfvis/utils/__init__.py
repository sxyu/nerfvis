try:
    from ._rotation import Rotation
except:
    from scipy.spatial.transform import Rotation # If cython not available, requires scipy
