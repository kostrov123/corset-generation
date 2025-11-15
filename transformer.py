import os

import trimesh
import numpy as np
import pandas as pd
import vedo
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

from numpy import expand_dims


class Model3D:
    def __init__(self, mesh: trimesh.Trimesh, pitch=1):
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError()
        self.mesh = mesh
        self.voxels = mesh.voxelized(pitch=pitch)


class Transformer:
    def __init__(self, image_size=(256, 256),
                 xyz_compression=np.array([2, 2, 4]),
                 y_shift=1e-3
                 ):
        self.image_size = image_size
        self.xyz_compression = xyz_compression
        self.y_shift = y_shift

    @staticmethod
    def __rotation_matrix(a):
        """Матрица вращений"""
        return np.array(
            [
                [np.cos(a), -np.sin(a), 0],
                [np.sin(a), np.cos(a), 0],
                [0, 0, 1]
            ]
        )

    def points2projection(self, points: np.ndarray, angle: float) -> np.ndarray:
        __points = (
            points.dot(self.__rotation_matrix(angle))
            / self.xyz_compression
        )
        df_points = pd.DataFrame(
            {
                'X': __points[:, 0].round().astype(int),
                'Y': __points[:, 1],
                'Z': __points[:, 2].round().astype(int)
            }
        )

        df_pixels = (
            df_points
            .query('Y > 0')
            .groupby(['X', 'Z'])['Y'].mean()
            .reset_index()
        )
        data = np.zeros(shape=self.image_size, dtype=float)
        data[
            -df_pixels['Z'] + self.image_size[0] // 2,
            -df_pixels['X'] + self.image_size[1] // 2
        ] = df_pixels['Y'] + self.y_shift
        return data

    def projection2points(self, data: np.ndarray, angle: float):
        assert data.shape == self.image_size
        z, x = np.where(data >= self.y_shift)
        points = np.vstack(
            (
                -x + self.image_size[1] // 2,
                data[z, x] - self.y_shift,
                -z + self.image_size[0] // 2
            )
        ).T.dot(self.__rotation_matrix(-angle)) * self.xyz_compression

        return points

    @classmethod
    def points2stl(cls, points, r_points=1, r_neighbours=3, neighbours=10, mesh_radius=3):
        """

        :param points:
        :param r_points:
        :param r_neighbours:
        :param neighbours:
        :param mesh_radius:
        :return:
        """
        pts = vedo.Points(points, r=r_points).remove_outliers(r_neighbours, neighbors=neighbours)
        #pts = removeOutliers(
        #    points=vedo.Points(points, r=r_points),
        #    neighbors=neighbours,
        #    radius=r_neighbours
        #)
        mesh = vedo.recoSurface(pts, radius=mesh_radius)
        return mesh


class NNModel:

    def __init__(self, path: str,
                 transformer: Transformer,
                 n_slices: int = 4
                 ):
        self.model = load_model(path, compile=False)
        self.transformer = transformer
        self.N_SLICES = n_slices

    @classmethod
    def image2data(cls, image):
        img = np.array(image)
        pixels = np.stack((img,) * 3, axis=-1)
        pixels = (pixels - 127.5) / 127.5
        pixels = expand_dims(pixels, 0)
        return pixels

    def predict(self, input_stl: Model3D,
                voxel_size: int = 1,
                seek_n_neighbours: int = 15,
                seek_radius: int = 15,
                reco_radius: int = 7
                ):
        # Считаем кол-во точек в исходной модели
        num_p = input_stl.mesh.vertices.shape[0]

        # Нарезаем исходную модель на проекции
        input_slices = [
            self.transformer.points2projection(
                points=input_stl.voxels.points,
                angle=2 * np.pi * i / self.N_SLICES
            )
            for i in range(self.N_SLICES)
        ]

        # Трансформируем проекции человека в проекции корсета
        output_slices = [
            (self.model.predict(self.image2data(image=image))[0]) * 127.5 + 127.5
            for image in input_slices
        ]

        # Собираем картинки в облако точек
        points = np.vstack([
            self.transformer.projection2points(
                data=output_slices[i][:, :, 0],
                angle=2 * np.pi * i / self.N_SLICES
            )
            for i in range(self.N_SLICES)
        ])

        # Прореживаем точки
        idx = np.random.randint(low=0, high=points.shape[0], size=num_p * 2)
        points = points[idx, :]

        # Финализируем черех пакет vedo
        points = vedo.Points(points, r=voxel_size)

        # Сглаживаем точки, длительная операция
        points.smoothMLS2D(f=0.8)
        points.remove_outliers()
        # Удаляем выбросы
        points = removeOutliers(
            points=points,
            neighbors=seek_n_neighbours,
            radius=seek_radius
        )
        points.clean(tol=0.005)

        # Натягиваем поверхность на точки
        mesh = vedo.recoSurface(points, radius=reco_radius)

        return mesh

