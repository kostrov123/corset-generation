import os
import numpy as np
import vedo
from PIL import Image
from points_projector import PointsProjector


class ImageModel:
    """
    Предсказывает mesh из предсказаных карт высот.
    """
    def __init__(self,
                 transformer: PointsProjector,
                 n_slices: int = 4
                 ):
        """
        @param transformer: проектор точек.
        @param n_slices: число срезов для построения модели.
        """
        self.points_projector = transformer
        self.n_slices = n_slices

    def predict(self, output_slices_paths: [str],
                num_p: int = 18345,
                voxel_size: int = 1,
                seek_n_neighbours: int = 15,
                seek_radius: int = 15,
                reco_radius: int = 7
                ):

        # Собираем картинки в облако точек
        output_slices = [[np.asarray(Image.open(path)), os.path.basename(path).split('.')[-2].split('_')[-1]] for path in output_slices_paths]
        points = np.vstack([
            self.points_projector.projection2points(
                data=output_slices[i][0][:, :, 0] / 2,
                angle=2 * np.pi * float(output_slices[i][1]) / 360
            )
            for i in range(self.n_slices)
        ])

        # Прореживаем точки
        idx = np.random.randint(low=0, high=points.shape[0], size=num_p * 2)
        points = points[idx, :]

        # Финализируем черех пакет vedo
        points = vedo.Points(points, r=voxel_size)

        # Сглаживаем точки, длительная операция
        points.smooth_mls_2d(f=0.8, radius=seek_radius)

        # Удаляем выбросы
        points = points.remove_outliers(
            neighbors=seek_n_neighbours,
            radius=seek_radius
        )

        points.clean()#tol=0.005) # У этой функции нет аргументов в новой версии.

        # Натягиваем поверхность на точки
        mesh = points.reconstruct_surface(radius=reco_radius)
        return mesh