# Example to add an image
from nerfvis import scene
import numpy as np
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation

bulldozer = imageio.imread("bulldozer.jpg")
R = Rotation.from_rotvec(np.pi / 2 * np.array([0, 1, 0])).as_matrix()
t = np.random.randn(3) * 0.1
f = 358 / 800 * 1111
Z = 1.0
scene.set_opencv()

scene.add_cube("My cube/0", translation=R[:, 2] * 2, color=[1, 0, 0])
scene.add_camera_frustum(
    "My camera/0",
    r=R[None],
    t=t[None],
    focal_length=f,
    image_width=bulldozer.shape[1],
    image_height=bulldozer.shape[0],
    z=Z,
)
scene.add_image(
    "My bulldozer image", bulldozer, r=R, t=t, focal_length=f, z=Z
)
scene.display(port=6006)
