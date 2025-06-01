import numpy as np
import cupy as cp

class GPUMemoryManager:
    """Handle efficient GPU memory allocation and transfer"""
    
    def __init__(self):
        self.pinned_memory = {}
        self.device_memory = {}
        self.use_gpu = cp.cuda.is_available()
        
    def allocate(self, name, shape, dtype=cp.float32):
        """Allocate memory on GPU with a given name"""
        if not self.use_gpu:
            return np.zeros(shape, dtype=np.dtype(dtype))
            
        try:
            if name in self.device_memory:
                # Only reallocate if shape or dtype changed
                existing = self.device_memory[name]
                if existing.shape != shape or existing.dtype != dtype:
                    self.device_memory[name] = cp.zeros(shape, dtype=dtype)
            else:
                self.device_memory[name] = cp.zeros(shape, dtype=dtype)
            
            return self.device_memory[name]
        except cp.cuda.memory.OutOfMemoryError:
            print(f"GPU memory exceeded when allocating {name} - falling back to CPU")
            return np.zeros(shape, dtype=np.dtype(dtype))
        
    def transfer_to_gpu(self, name, data):
        """Transfer data to GPU efficiently"""
        if not self.use_gpu:
            return data
            
        try:
            # Handle different input types
            if isinstance(data, list) and all(isinstance(item, np.ndarray) for item in data):
                # List of arrays
                return [self._transfer_single_to_gpu(f"{name}_{i}", arr) for i, arr in enumerate(data)]
            elif hasattr(data, '__dict__'):
                # Object with attributes - extract numpy arrays
                result = {}
                for attr_name, attr_value in data.__dict__.items():
                    if isinstance(attr_value, np.ndarray):
                        result[attr_name] = self._transfer_single_to_gpu(f"{name}_{attr_name}", attr_value)
                    elif isinstance(attr_value, list) and all(isinstance(item, np.ndarray) for item in attr_value):
                        result[attr_name] = [self._transfer_single_to_gpu(
                            f"{name}_{attr_name}_{i}", arr) for i, arr in enumerate(attr_value)]
                return result
            else:
                # Single array
                return self._transfer_single_to_gpu(name, data)
        except cp.cuda.memory.OutOfMemoryError:
            print(f"GPU memory exceeded when transferring {name} - falling back to CPU")
            return data
    
    def _transfer_single_to_gpu(self, name, data):
        """Transfer a single array to GPU"""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")
            
        # Use pinned memory for faster transfers
        if name not in self.pinned_memory or self.pinned_memory[name].shape != data.shape:
            self.pinned_memory[name] = cp.zeros_pinned(data.shape, dtype=data.dtype)
        
        # Copy to pinned memory
        cp.copyto(self.pinned_memory[name], data)
        
        # Allocate GPU memory if needed
        if name not in self.device_memory or self.device_memory[name].shape != data.shape:
            self.device_memory[name] = cp.zeros(data.shape, dtype=data.dtype)
            
        # Copy from pinned memory to GPU
        self.device_memory[name].set(self.pinned_memory[name])
        
        return self.device_memory[name]
        
    def transfer_to_cpu(self, name):
        """Transfer data from GPU to CPU efficiently"""
        if not self.use_gpu or name not in self.device_memory:
            if isinstance(name, str) and name.startswith('list:'):
                # Handle list case
                list_name = name[5:]  # Remove 'list:' prefix
                result = []
                i = 0
                while f"{list_name}_{i}" in self.device_memory:
                    result.append(self._transfer_single_to_cpu(f"{list_name}_{i}"))
                    i += 1
                return result
            elif isinstance(name, str) and name.startswith('obj:'):
                # Handle object case
                obj_name = name[4:]  # Remove 'obj:' prefix
                result = {}
                for key in list(self.device_memory.keys()):
                    if key.startswith(f"{obj_name}_"):
                        attr_name = key[len(obj_name)+1:]
                        result[attr_name] = self._transfer_single_to_cpu(key)
                return result
            
            if isinstance(name, str) and name in self.device_memory:
                return self._transfer_single_to_cpu(name)
                
            raise KeyError(f"No GPU data found for {name}")
            
    def _transfer_single_to_cpu(self, name):
        """Transfer a single array from GPU to CPU"""
        if name not in self.device_memory:
            raise KeyError(f"No GPU data found for {name}")
            
        # Use pinned memory for faster transfers
        if name not in self.pinned_memory or self.pinned_memory[name].shape != self.device_memory[name].shape:
            self.pinned_memory[name] = cp.zeros_pinned(
                self.device_memory[name].shape, 
                dtype=self.device_memory[name].dtype)
        
        # Copy from GPU to pinned memory
        self.device_memory[name].get(out=self.pinned_memory[name])
        
        # Convert to regular numpy array
        return np.array(self.pinned_memory[name])
    
    def clear(self, name=None):
        """Clear memory for a specific name or all if name is None"""
        if name is None:
            self.device_memory = {}
            self.pinned_memory = {}
            cp.get_default_memory_pool().free_all_blocks()
        else:
            if name in self.device_memory:
                del self.device_memory[name]
            if name in self.pinned_memory:
                del self.pinned_memory[name]