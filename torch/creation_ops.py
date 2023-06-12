import torch
import numpy as np

# tensor
# Constructs a tensor with no autograd history (also known as a "leaf tensor", see Autograd mechanics) by copying data.
data = [1, 2, 3]
tensor = torch.tensor(data)
print(tensor)

# sparse_coo_tensor
# Constructs a sparse tensor in COO(rdinate) format with specified values at the given indices.
values = torch.tensor([1, 2, 3])
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
shape = (3, 3)
sparse_coo_tensor = torch.sparse_coo_tensor(indices, values, shape)
print(sparse_coo_tensor)

# sparse_csr_tensor
# Constructs a sparse tensor in CSR (Compressed Sparse Row) with specified values at the given row_indices and col_indices.
values = torch.tensor([1, 2, 3])
row_indices = torch.tensor([0, 1, 1])
col_indices = torch.tensor([2, 0, 2])
shape = (2, 3)
sparse_csr_tensor = torch.sparse_csr_tensor(row_indices, col_indices, values, shape)
print(sparse_csr_tensor)

# sparse_csc_tensor
# Constructs a sparse tensor in CSC (Compressed Sparse Column) with specified values at the given col_indices and row_indices.
values = torch.tensor([1, 2, 3])
col_indices = torch.tensor([2, 0, 2])
row_indices = torch.tensor([0, 1, 1])
shape = (3, 2)
sparse_csc_tensor = torch.sparse_csc_tensor(row_indices, col_indices, values, shape)
print(sparse_csc_tensor)

# sparse_bsc_tensor
# Constructs a sparse tensor in BSC (Block Compressed Sparse Column) with specified 2-dimensional blocks at the given ccol_indices and row_indices.
#values = torch.tensor([[1, 2], [3, 4], [5, 6]])
#col_indices = torch.tensor([2, 0, 2])
#row_indices = torch.tensor([0, 1, 1])
#blocksize = (2, 2)
#shape = (6, 4)
#sparse_bsc_tensor = torch.sparse_bsc_tensor(col_indices, row_indices, values, blocksize, shape)
#print(sparse_bsc_tensor)


# sparse_bsc_tensor
# Constructs a sparse tensor in BSC (Block Compressed Sparse Column) with specified 2-dimensional blocks at the given col_indices and row_indices.
#values = torch.tensor([[1, 2], [3, 4], [5, 6]])
#col_indices = torch.tensor([2, 0, 2])
#row_indices = torch.tensor([0, 1, 1])
#blocksize = (2, 2)
#shape = (6, 4)
#sparse_bsc_tensor = torch.sparse_bsc_tensor(row_indices, col_indices, values, blocksize, shape)
#print(sparse_bsc_tensor)

# asarray
# Converts obj to a tensor.
numpy_array = np.array([1, 2, 3])
asarray_tensor = torch.asarray(numpy_array)
print(asarray_tensor)

# as_tensor
# Converts data into a tensor, sharing data and preserving autograd history if possible.
data = [1, 2, 3]
as_tensor_tensor = torch.as_tensor(data)
print(as_tensor_tensor)

# as_strided
# Create a view of an existing torch.Tensor input with specified size, stride and storage_offset.
input_tensor = torch.tensor([1, 2, 3, 4])
as_strided_tensor = torch.as_strided(input_tensor, size=(2, 2), stride=(2, 1), storage_offset=0)
print(as_strided_tensor)

# from_numpy
# Creates a Tensor from a numpy.ndarray.
numpy_array = np.array([1, 2, 3])
from_numpy_tensor = torch.from_numpy(numpy_array)
print(from_numpy_tensor)

# from_dlpack
# Converts a tensor from an external library into a torch.Tensor.
# Note: The usage of from_dlpack depends on the specific external library. Example code can vary.

# frombuffer
# Creates a 1-dimensional Tensor from an object that implements the Python buffer protocol.
buffer_obj = bytearray([1, 2, 3])
frombuffer_tensor = torch.frombuffer(buffer_obj, dtype=torch.uint8)
print(frombuffer_tensor)

# zeros
# Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
size = (2, 3)
zeros_tensor = torch.zeros(size)
print(zeros_tensor)

# zeros_like
# Returns a tensor filled with the scalar value 0, with the same size as input.
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
zeros_like_tensor = torch.zeros_like(input_tensor)
print(zeros_like_tensor)

# ones
# Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
size = (2, 3)
ones_tensor = torch.ones(size)
print(ones_tensor)

# ones_like
# Returns a tensor filled with the scalar value 1, with the same size as input.
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
ones_like_tensor = torch.ones_like(input_tensor)
print(ones_like_tensor)

# arange
# Returns a 1-D tensor of size ⌈(end - start) / step⌉ with values from the interval [start, end) taken with common difference step beginning from start.
start = 0
end = 5
step = 1
arange_tensor = torch.arange(start, end, step)
print(arange_tensor)

# range
# Returns a 1-D tensor of size ⌊(end - start) / step⌋ + 1 with values from start to end with step step.
start = 0
end = 5
step = 1
range_tensor = torch.range(start, end, step)
print(range_tensor)

# linspace
# Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
start = 0
end = 1
steps = 5
linspace_tensor = torch.linspace(start, end, steps)
print(linspace_tensor)

# logspace
# Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base base.
base = 2
start = 0
end = 3
steps = 4
logspace_tensor = torch.logspace(start, end, steps, base)
print(logspace_tensor)

# eye
# Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
n = 3
eye_tensor = torch.eye(n)
print(eye_tensor)

# empty
# Returns a tensor filled with uninitialized data.
size = (2, 3)
empty_tensor = torch.empty(size)
print(empty_tensor)

# empty_like
# Returns an uninitialized tensor with the same size as input.
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
empty_like_tensor = torch.empty_like(input_tensor)
print(empty_like_tensor)

# empty_strided
# Creates a tensor with the specified size and stride and filled with undefined data.
#size = (2, 3)
#stride = (3, 1)
#storage_offset = 0
#empty_strided_tensor = torch.empty_strided(size, stride, storage_offset)
#print(empty_strided_tensor)

# full
# Creates a tensor of size size filled with fill_value.
size = (2, 3)
fill_value = 5
full_tensor = torch.full(size, fill_value)
print(full_tensor)

# full_like
# Returns a tensor with the same size as input filled with fill_value.
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
fill_value = 5
full_like_tensor = torch.full_like(input_tensor, fill_value)
print(full_like_tensor)

# quantize_per_tensor
# Converts a float tensor to a quantized tensor with given scale and zero point.
#float_tensor = torch.tensor([0.1, 0.2, 0.3])
#scale = 0.1
#zero_point = 2
#quantize_per_tensor_tensor = torch.quantize_per_tensor(float_tensor, scale, zero_point)
#print(quantize_per_tensor_tensor)

# quantize_per_channel
# Converts a float tensor to a per-channel quantized tensor with given scales and zero points.
float_tensor = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
scales = torch.tensor([0.1, 0.2, 0.3])
zero_points = torch.tensor([2, 3, 4])
quantize_per_channel_tensor = torch.quantize_per_channel(float_tensor, scales, zero_points, 1, torch.qint8)
print(quantize_per_channel_tensor)

# dequantize
# Returns an fp32 Tensor by dequantizing a quantized Tensor.
#quantized_tensor = torch.tensor([1, 2, 3], dtype=torch.qint8)
#dequantize_tensor = torch.dequantize(quantized_tensor)
#print(dequantize_tensor)

# complex
# Constructs a complex tensor with its real part equal to real and its imaginary part equal to imag.
#real = torch.tensor([1, 2, 3])
#imag = torch.tensor([4, 5, 6])
#complex_tensor = torch.complex(real, imag)
#print(complex_tensor)

# polar
# Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value abs and angle angle.
#abs = torch.tensor([1, 2, 3])
#angle = torch.tensor([0, np.pi/2, np.pi])
#polar_tensor = torch.polar(abs, angle)
#print(polar_tensor)

# heaviside
# Computes the Heaviside step function for each element in input.
input_tensor = torch.tensor([-1, 0, 1])
heaviside_tensor = torch.heaviside(input_tensor, values=torch.tensor([0]))
print(heaviside_tensor)

