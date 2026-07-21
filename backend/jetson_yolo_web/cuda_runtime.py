import ctypes


class CudaRuntimeError(RuntimeError):
    pass


class CudaRuntime(object):
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self):
        self.lib = _load_cudart()
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def malloc(self, size):
        pointer = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(size)), "cudaMalloc")
        return pointer.value

    def free(self, pointer):
        if pointer:
            self._check(self.lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def memcpy_htod(self, device_pointer, host_array):
        host_pointer = host_array.ctypes.data_as(ctypes.c_void_p)
        self._check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(device_pointer),
                host_pointer,
                ctypes.c_size_t(host_array.nbytes),
                self.cudaMemcpyHostToDevice,
            ),
            "cudaMemcpyHostToDevice",
        )

    def memcpy_dtoh(self, host_array, device_pointer):
        host_pointer = host_array.ctypes.data_as(ctypes.c_void_p)
        self._check(
            self.lib.cudaMemcpy(
                host_pointer,
                ctypes.c_void_p(device_pointer),
                ctypes.c_size_t(host_array.nbytes),
                self.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpyDeviceToHost",
        )

    def _check(self, code, operation):
        if code != 0:
            message = self.lib.cudaGetErrorString(code)
            if message:
                message = message.decode("utf-8", "replace")
            else:
                message = "CUDA error %s" % code
            raise CudaRuntimeError("%s failed: %s" % (operation, message))


def _load_cudart():
    names = ["libcudart.so", "libcudart.so.10.2", "/usr/local/cuda/lib64/libcudart.so"]
    last_error = None
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            last_error = exc
    raise CudaRuntimeError("Unable to load CUDA runtime: %s" % last_error)
