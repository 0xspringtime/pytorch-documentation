import torch

# adjoint
# Returns a view of the tensor conjugated and with the last two dimensions transposed.
tensor = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]])
adjoint_tensor = tensor.adjoint()
print(adjoint_tensor)

# argwhere
# Returns a tensor containing the indices of all non-zero elements of input.
input_tensor = torch.tensor([[0, 1, 0], [2, 0, 3]])
argwhere_tensor = torch.argwhere(input_tensor)
print(argwhere_tensor)

# cat
# Concatenates the given sequence of seq tensors in the given dimension.
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
cat_tensor = torch.cat((tensor1, tensor2), dim=0)
print(cat_tensor)

# conj
# Returns a view of input with a flipped conjugate bit.
tensor = torch.tensor([[1-2j, 3+4j], [5+6j, 7+8j]])
conj_tensor = tensor.conj()
print(conj_tensor)

# chunk
# Attempts to split a tensor into the specified number of chunks.
tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7])
chunked_tensor = torch.chunk(tensor, chunks=3, dim=0)
print(chunked_tensor)

# dsplit
# Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections.
#tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]],
#                      [[7, 8, 9], [10, 11, 12]]])
#dsplit_tensors = torch.dsplit(tensor, indices_or_sections=3)
#for t in dsplit_tensors:
#    print(t)

# column_stack
# Creates a new tensor by horizontally stacking the tensors in tensors.
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
column_stacked_tensor = torch.column_stack((tensor1, tensor2))
print(column_stacked_tensor)

# dstack
# Stack tensors in sequence depthwise (along third axis).
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
dstacked_tensor = torch.dstack((tensor1, tensor2))
print(dstacked_tensor)


# gather
# Gathers values along an axis specified by dim.
tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
index = torch.tensor([[0, 1], [2, 0]])
gathered_tensor = torch.gather(tensor, dim=0, index=index)
print(gathered_tensor)

# index_select
# Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
index = torch.tensor([0, 2])
selected_tensor = torch.index_select(tensor, dim=0, index=index)
print(selected_tensor)

# nonzero
# Returns a tensor containing the indices of all non-zero elements of input.
tensor = torch.tensor([[0, 1, 0], [2, 0, 3]])
nonzero_tensor = torch.nonzero(tensor)
print(nonzero_tensor)

# reshape
# Returns a tensor with the same data and number of elements as input, but with the specified shape.
tensor = torch.tensor([[1, 2], [3, 4]])
reshaped_tensor = torch.reshape(tensor, shape=(1, 4))
print(reshaped_tensor)

# squeeze
# Returns a tensor with all specified dimensions of input of size 1 removed.
tensor = torch.tensor([[[1]]])
squeezed_tensor = torch.squeeze(tensor)
print(squeezed_tensor)

# stack
# Concatenates a sequence of tensors along a new dimension.
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
stacked_tensor = torch.stack((tensor1, tensor2), dim=0)
print(stacked_tensor)

# transpose
# Returns a tensor that is a transposed version of input.
tensor = torch.tensor([[1, 2], [3, 4]])
transposed_tensor = torch.transpose(tensor, dim0=0, dim1=1)
print(transposed_tensor)

# unsqueeze
# Returns a new tensor with a dimension of size one inserted at the specified position.
tensor = torch.tensor([1, 2, 3])
unsqueeze_tensor = torch.unsqueeze(tensor, dim=1)
print(unsqueeze_tensor)

# where
# Return a tensor of elements selected from either input or other, depending on condition.
condition = torch.tensor([[True, False], [False, True]])
input_tensor = torch.tensor([[1, 2], [3, 4]])
other_tensor = torch.tensor([[5, 6], [7, 8]])
where_tensor = torch.where(condition, input_tensor, other_tensor)
print(where_tensor)

