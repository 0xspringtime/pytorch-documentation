import torch

# no_grad
# Context-manager that disables gradient calculation.
with torch.no_grad():
    x = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    y = x * 2
    print(y.requires_grad)  # False

# enable_grad
# Context-manager that enables gradient calculation.
with torch.enable_grad():
    x = torch.tensor([1, 2, 3])
    y = x * 2
    print(y.requires_grad)  # True

# set_grad_enabled
# Context-manager that sets gradient calculation on or off.
with torch.set_grad_enabled(False):
    x = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    y = x * 2
    print(y.requires_grad)  # False

# is_grad_enabled
# Returns True if grad mode is currently enabled.
grad_enabled = torch.is_grad_enabled()
print(grad_enabled)  # True

# inference_mode
# Context-manager that enables or disables inference mode
with torch.inference_mode():
    x = torch.tensor([1, 2, 3], dtype=torch.float32,requires_grad=True)
    y = x * 2
    print(y.requires_grad)  # False

# is_inference_mode_enabled
# Returns True if inference mode is currently enabled.
inference_mode_enabled = torch.is_inference_mode_enabled()
print(inference_mode_enabled)  # False

