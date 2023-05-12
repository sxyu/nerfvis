# Example to add an image
from nerfvis import scene
import numpy as np
import imageio.v2 as imageio

bulldozer = imageio.imread('bulldozer.jpg')
scene.add_cube('My cube/0', translation=[0,0,0])
scene.add_image('My bulldozer image',
                bulldozer,
                r = np.eye(3),
                t = np.zeros(3),
                focal_length=358 / 800 * 1111,
                z=1.0,
                opengl=False
            )
scene.set_opencv()
scene.display(port=6006)
