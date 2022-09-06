# Load and show a plenoctree checkpoint
from nerfvis import scene

# Download from
# https://drive.google.com/drive/u/1/folders/1vGXEjb3yhbClrZH1vLdl2iKtowfinWOg

scene.set_title("Lego Bulldozer using nerfvis")
scene.add_volume_from_npz('Lego', "lego.npz")

scene.display(port=6006)
