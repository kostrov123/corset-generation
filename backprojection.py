import glob
import os
import numpy as np
import trimesh

from image_model import ImageModel
from points_projector import PointsProjector

IMAGE_SIZE = (256, 256)
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECTION_DIR = os.path.join(CURRENT_PATH, "../eval/")
RAW_DATA_DIR = os.path.join(CURRENT_PATH, "../data_new/raw_data/")

projection_filename_mask = "{client_id}_{stl_type}_{angle}.jpg"

N_SLICES = 8

# Для ускорения расчета моделей
USE_PRECALCULATED_POINTS_COUNT = True

class Model3D:
    def __init__(self, mesh: trimesh.Trimesh, pitch=1):
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError()
        self.mesh = mesh
        self.voxels = mesh.voxelized(pitch=pitch)


def project_to_stl():
    predicted_projection = glob.iglob(PROJECTION_DIR + "**/*.png", recursive=True)
    predicted_projection = list(predicted_projection)
    predicted_projection = [(os.path.basename(f).split('_')[0], f) for f in predicted_projection]
    predicted_projection_keys = np.unique([p[0] for p in predicted_projection])
    images_model = ImageModel(PointsProjector(), 8)
    for key in predicted_projection_keys:
        predicted_projection_group = [p[1] for p in predicted_projection if p[0] == key]
        print(predicted_projection_group)
        if not USE_PRECALCULATED_POINTS_COUNT:
            input_mesh = trimesh.load_mesh(os.path.join(RAW_DATA_DIR, f'1.{key}.1.stl'))
            input_model_3d = Model3D(mesh=input_mesh)
            num_p = input_model_3d.mesh.vertices.shape[0]

        num_p = 18345 if USE_PRECALCULATED_POINTS_COUNT else input_model_3d.mesh.vertices.shape[0]
        mesh = images_model.predict(predicted_projection_group, reco_radius=15)
        mesh.write(f'output_{key}.stl')


if __name__ == "__main__":
    project_to_stl()