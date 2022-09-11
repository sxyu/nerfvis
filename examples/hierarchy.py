# A basic dumb scene to demonstrate hierarchy in the left side tree view
from nerfvis import scene
import numpy as np

scene.add_cube('My cube/0', translation=[2,0,0])
scene.add_cube('My cube/1', color=[1,0,0])
scene.add_sphere('My sphere/0/0', color=[0,1,0], translation=[0,2,0])
scene.add_sphere('My sphere/1', color=[0,0,1], translation=[0,0,2])

# Randomly throw in a volume for varienty
#  density = 1.0 / (np.linalg.norm((np.mgrid[:127, :127, :127].transpose(1, 2, 3, 0) - 63.5) / 64.0,
#                           axis=-1) + 1e-5)
density = np.full((128, 128, 128), fill_value=1.5, dtype=np.float32)
color = np.full(list(density.shape) + [3], fill_value=0.5, dtype=np.float32)
scene.add_volume('My volume', density, color,
                 scale=2.0, translation=[-2, 0, 0])
                 #  density_threshold=0.0)
scene.display(port=6006)
