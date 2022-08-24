.. nerfvis documentation master file, created by
   sphinx-quickstart on Wed Sep 15 10:48:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nerfvis's documentation!
===================================

This is a PyTorch NeRF visualization utility based on PlenOctrees, as well as a 
general web-based 3D visualization library.

Install with :code:`pip install nerfvis`. This is a pure-Python library. The base library has no dependencies except for :code:`numpy`.
Adding meshes from a file requires :code:`trimesh`. 
Adding NeRF requires :code:`torch, svox, tqdm, scipy`.

Basic usage example:

.. code:: python

    import nerfvis
    scene = nerfvis.Scene("My title")
    scene.add_cube("Cube/1", color=[1.0, 0.0, 0.0], translation=[-1.0, -1.0, 0.0])
    scene.add_wireframe_cube("Cube/2", color=[1.0, 0.0, 0.0],
                             translation=[-1.0, -1.0, 0.0], scale=2.0)
    scene.add_axes()
    scene.set_nerf(nerf_func, center=[0.0, 0.0, 0.0], radius=1.5, use_dirs=True) # Optional
    scene.display()  # Serves at localhost:8888 or first port available after that
    scene.export("folder") # Exports sources to location
    scene.embed() # Embed in IPython notebook

Scene: holds objects. If you only use one scene, feel free to use :code:`from nerfvis import scene` instead.

Names separated by :code:`/` will be collapsed into a tree in the layers panel in the output.


Other common functions include `add_camera_frustum`, `add_mesh`, `add_line`, `add_lines`,
`add_image`

Please see :ref:`nerfvis` for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   nerfvis

