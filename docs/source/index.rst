.. nerfvis documentation master file, created by
   sphinx-quickstart on Wed Sep 15 10:48:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nerfvis's documentation!
===================================

This is an experimental PyTorch NeRF visualization utility based on PlenOctrees.

Install with :code:`pip install nerfvis`. This is a pure-Python library. The base library has no dependencies except for :code:`numpy`.
Adding meshes from a file requires :code:`trimesh`. 
Adding NeRF requires :code:`torch, svox, tqdm, scipy`.

Basic usage example:

.. code:: python

    import nerfvis
    scene = nerfvis.Scene("My title")
    scene.add_cube("Cube1", color=[1.0, 0.0, 0.0], translation=[-1.0, -1.0, 0.0])
    scene.add_axes()
    scene.set_nerf(nerf_func, center=[0.0, 0.0, 0.0], radius=1.5, use_dirs=True)
    scene.display(port=8889)
    # Tries to open the scene in your browser
    # (you may have to forward the port and enter localhost:8889 manually if over ssh)

Please see :ref:`nerfvis` for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   nerfvis

