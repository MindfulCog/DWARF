import numpy as np
from typing import Union

class Vector3:
    """Optimized 3D vector class with NumPy backend"""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._data = np.array([x, y, z], dtype=np.float32)
    
    @property
    def x(self) -> float:
        return float(self._data[0])
    
    @x.setter
    def x(self, value: float):
        self._data[0] = value
    
    @property
    def y(self) -> float:
        return float(self._data[1])
    
    @y.setter
    def y(self, value: float):
        self._data[1] = value
    
    @property
    def z(self) -> float:
        return float(self._data[2])
    
    @z.setter
    def z(self, value: float):
        self._data[2] = value
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        result = Vector3()
        result._data = self._data + other._data
        return result
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        result = Vector3()
        result._data = self._data - other._data
        return result
    
    def __mul__(self, scalar: float) -> 'Vector3':
        result = Vector3()
        result._data = self._data * scalar
        return result
    
    def __rmul__(self, scalar: float) -> 'Vector3':
        return self * scalar
    
    def __truediv__(self, scalar: float) -> 'Vector3':
        result = Vector3()
        result._data = self._data / scalar
        return result
    
    def dot(self, other: 'Vector3') -> float:
        return float(np.dot(self._data, other._data))
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        result = Vector3()
        result._data = np.cross(self._data, other._data)
        return result
    
    def magnitude(self) -> float:
        return float(np.linalg.norm(self._data))
    
    def normalized(self) -> 'Vector3':
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector3(0, 0, 0)
    
    def __str__(self) -> str:
        return f"Vector3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def __getitem__(self, index: int) -> float:
        return float(self._data[index])
    
    def __setitem__(self, index: int, value: float):
        self._data[index] = value
    
    def to_numpy(self) -> np.ndarray:
        return self._data.copy()