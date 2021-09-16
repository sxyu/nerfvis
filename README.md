NeRF visualization library using PlenOctrees, under construction

`pip install nerfvis`

Docs will be at: https://nerfvis.readthedocs.org

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
