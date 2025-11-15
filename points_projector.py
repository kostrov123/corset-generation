import numpy as np


class PointsProjector:
    """
    Отдельный простой класс для отображения карт высот в точки mesh'a.
    """
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
