import os
import numpy as np
import trimesh
import pickle
from PIL import Image
from tqdm import tqdm
import glob

from transformer import Transformer, Model3D

stl_type_dict = {
    "1": "human",
    "2": "corset",
    "2.1": "corset"
}

IMAGE_SIZE = (256, 256)
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(CURRENT_PATH, "../data_new/raw_data/")
VOXELS_DIR = os.path.join(CURRENT_PATH, "../data_new/voxels/")
SLICES_DIR = os.path.join(
    CURRENT_PATH,
    "../data/projections/{height}_{width}/".format(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])
)

voxel_filename_mask = "{client_id}_{stl_type}.pkl"
projection_filename_mask = "{client_id}_{stl_type}_{angle}.jpg"

N_SLICES = 8


def project_stl_to_images():
    os.makedirs(SLICES_DIR, exist_ok=True)
    os.makedirs(VOXELS_DIR, exist_ok=True)

    for filepath in tqdm(glob.iglob(RAW_DATA_DIR + "**/*.stl", recursive=True)):
        try:
            filename = os.path.basename(filepath)

            transformer = Transformer(image_size=IMAGE_SIZE)
            if "_" in filename:
                client_id, stl_type_id = filename.replace(".stl", "").split("_")
            else:
                _, client_id, stl_type_id, _ = filename.replace(",", ".").split(".")

            stl_type = stl_type_dict[stl_type_id]

            voxel_path = os.path.join(
                VOXELS_DIR,
                voxel_filename_mask.format(client_id=client_id, stl_type=stl_type)
            )

            try:
                model_3d = pickle.load(open(voxel_path, 'rb'))
            except FileNotFoundError:
                mesh = trimesh.load_mesh(filepath)
                model_3d = Model3D(mesh=mesh)
                pickle.dump(model_3d, open(voxel_path, 'wb'))

            for i in range(N_SLICES):
                angle = 2 * np.pi * i / N_SLICES
                projection = transformer.points2projection(points=model_3d.voxels.points, angle=angle).astype(np.uint8)
                img = Image.fromarray(projection)

                projection_path = os.path.join(
                    SLICES_DIR,
                    projection_filename_mask.format(
                        client_id=client_id, stl_type=stl_type, angle=int(i * 360 / N_SLICES)
                    )
                )
                img.save(projection_path)
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    project_stl_to_images()