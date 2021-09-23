# nerf_pl Silica Visualization Example

![Screenshot](https://raw.githubusercontent.com/sxyu/nerfvis/master/examples/nerf_pl/img/silica.png)

This will visualize the silica model provided in the `nerf_pl` repo using `nerfvis`.

nerf_pl repo link: https://github.com/kwea123/nerf_pl

- A hardcoded bounding box will be used for the NeRF
- SH projection will be used to visualize the view-dependency (by simply setting `use_dirs=True` with default SH projection parameters)
- Default sigma threshold is used; to use the more advanced weight thresholding, you can try to grab `r, t, focal_length, image_width, image_height` from the dataset and pass them to `set_nerf`

Setup: download https://github.com/kwea123/nerf_pl to a directory `$ROOT_DIR` and install all the dependencies; or use your existing repo. To grab the silica checkpoints from Github:

```sh
ROOT_DIR=<enter nerf_pl git project root>
cp visualize.py $ROOT_DIR
mkdir $ROOT_DIR/silica
wget https://github.com/kwea123/nerf_pl/releases/download/v2.0.1/silica.ckpt -P $ROOT_DIR/silica
wget https://github.com/kwea123/nerf_pl/releases/download/v2.0.1/poses_bounds.npy -P $ROOT_DIR/silica
```

**Important**: For SH projection to work correctly, the network's output RGB must not pass through the sigmoid. You can very temporarily fix this by removing the `nn.Sigmoid()` in line 81 of `models/nerf.py`, in the definition of `self.rgb` of the `NeRF` class before running the demo below (note this will break training/eval).
A more permanent fix is by applying the sigmoid using `torch.sigmoid` in `NeRF.forward` instead, which you can control using a Boolean argument to `NeRF.forward`.

Then you can run the demo as

```sh
python $ROOT_DIR/visualize.py
```

If working locally (and port 8889 is not used), this will open the web browser to show the silica NeRF model.
If you're working over SSH, this should launch a server at `localhost:8889`. If you set up the SSH port forwarding, you can go to the forwarded port locally to view it.

