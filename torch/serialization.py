import torch

# save
# Saves an object to a disk file.
tensor = torch.tensor([1, 2, 3])
torch.save(tensor, 'saved_tensor.pt')

# load
# Loads an object saved with torch.save() from a file.
loaded_tensor = torch.load('saved_tensor.pt')
print(loaded_tensor)

