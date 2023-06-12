import torch

# get_num_threads
# Returns the number of threads used for parallelizing CPU operations
num_threads = torch.get_num_threads()
print(num_threads)

# set_num_threads
# Sets the number of threads used for intraop parallelism on CPU
torch.set_num_threads(4)  # Set the number of threads to 4

# get_num_interop_threads
# Returns the number of threads used for inter-op parallelism on CPU
num_interop_threads = torch.get_num_interop_threads()
print(num_interop_threads)

# set_num_interop_threads
# Sets the number of threads used for interop parallelism on CPU
torch.set_num_interop_threads(2)  # Set the number of inter-op threads to 2

