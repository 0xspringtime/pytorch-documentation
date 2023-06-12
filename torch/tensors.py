import torch

# is_tensor
# Returns True if obj is a PyTorch tensor.
tensor = torch.tensor([1, 2, 3])
print(torch.is_tensor(tensor))  # Output: True

# is_storage
# Returns True if obj is a PyTorch storage object.
storage = torch.Storage([1, 2, 3])
print(torch.is_storage(storage))  # Output: True

# is_complex
# Returns True if the data type of input is a complex data type i.e., one of torch.complex64, and torch.complex128.
complex_tensor = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
print(torch.is_complex(complex_tensor))  # Output: True

# is_conj
# Returns True if the input is a conjugated tensor, i.e. its conjugate bit is set to True.
conj_tensor = torch.conj(complex_tensor)
print(torch.is_conj(conj_tensor))  # Output: True

# is_floating_point
# Returns True if the data type of input is a floating point data type i.e., one of torch.float64, torch.float32, torch.float16, and torch.bfloat16.
float_tensor = torch.tensor([1.0, 2.0, 3.0])
print(torch.is_floating_point(float_tensor))  # Output: True

# is_nonzero
# Returns True if the input is a single element tensor which is not equal to zero after type conversions.
nonzero_tensor = torch.tensor([5])
print(torch.is_nonzero(nonzero_tensor))  # Output: True

# set_default_dtype
# Sets the default floating point dtype to d.
torch.set_default_dtype(torch.float64)

# get_default_dtype
# Get the current default floating point torch.dtype.
print(torch.get_default_dtype())  # Output: torch.float64

# set_default_device
# Sets the default torch.Tensor to be allocated on device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# set_default_tensor_type
# Sets the default torch.Tensor type to floating point tensor type t.
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

# numel
# Returns the total number of elements in the input tensor.
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor.numel())  # Output: 6

# set_printoptions
# Set options for printing.
torch.set_printoptions(precision=4)

# set_flush_denormal
# Disables denormal floating numbers on CPU.
torch.set_flush_denormal(True)

