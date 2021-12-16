NeRF visualization library using PlenOctrees, under construction

`pip install nerfvis`

Docs: https://nerfvis.readthedocs.org

Based on PlenOctrees: https://github.com/sxyu/plenoctrees

![Screenshot DTU skull](https://raw.githubusercontent.com/sxyu/nerfvis/master/img/skull.gif)

![Screenshot NeRF-- Drone](https://raw.githubusercontent.com/sxyu/nerfvis/master/img/youtube_drone.gif)


Example: see `examples/`; currently contains an example for nerf_pl:
![Basic silica low ](https://raw.githubusercontent.com/sxyu/nerfvis/master/img/silica_low.gif)

```python
import nerfvis
scene = nerfvis.Scene("My title")
scene.add_cube("Cube1", color=[1.0, 0.0, 0.0], translation=[-1.0, -1.0, 0.0])
scene.add_axes()
scene.set_nerf(nerf_func, center=[0.0, 0.0, 0.0], radius=1.5, use_dirs=True)
scene.display(port=8889)
# Tries to open the scene in your browser
# (you may have to forward the port and enter localhost:8889 manually if over ssh)
```

You can also add meshes, points, lines (see docs).
Note that each object e.g. cube, mesh, points, etc. must have a unique name to identify it
right now. You may programmatically generate this. They will show up in the *layers* pane
(top right of the html viewer)
**New in 0.0.6: Use / in the name e.g. cubes/red/cube001 to put meshes into folders in the layers pane.**


Please also `pip install torch svox tqdm scipy` for adding NeRF (`set_nerf`)
or `pip install trimesh` for using `add_mesh_from_file(path)`.

To add cameras (also used for scaling scene, initializing camera etc), use 
`add_camera_frustum(focal_length=.., image_width=.., image_height=.., z=..,  r=.., t=..)`


If you find  this useful please consider citing
```
@inproceedings{yu2021plenoctrees,
      title={{PlenOctrees} for Real-time Rendering of Neural Radiance Fields},
      author={Alex Yu and Ruilong Li and Matthew Tancik and Hao Li and Ren Ng and Angjoo Kanazawa},
      year={2021},
      booktitle={ICCV},
}
```
