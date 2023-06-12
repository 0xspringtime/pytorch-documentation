import torch

# abs
# Computes the absolute value of each element in input.
input_tensor = torch.tensor([-1, 2, -3])
abs_tensor = torch.abs(input_tensor)
print(abs_tensor)

# acos
# Computes the inverse cosine of each element in input.
input_tensor = torch.tensor([0.5, -0.5])
acos_tensor = torch.acos(input_tensor)
print(acos_tensor)

# acosh
# Returns a new tensor with the inverse hyperbolic cosine of the elements of input.
input_tensor = torch.tensor([1, 2, 3])
acosh_tensor = torch.acosh(input_tensor)
print(acosh_tensor)

# add
# Adds other, scaled by alpha, to input.
input_tensor = torch.tensor([1, 2, 3])
other_tensor = torch.tensor([4, 5, 6])
alpha = 1
add_tensor = torch.add(input_tensor, alpha, other_tensor)
print(add_tensor)

# addcdiv
# Performs the element-wise division of tensor1 by tensor2, multiplies the result by the scalar value and adds it to input.
#input_tensor = torch.tensor([1, 2, 3])
#tensor1 = torch.tensor([4, 5, 6])
#tensor2 = torch.tensor([2, 2, 2])
#value = 0.5
#addcdiv_tensor = torch.addcdiv(input_tensor, value, tensor1, tensor2)
#print(addcdiv_tensor)

# addcmul
# Performs the element-wise multiplication of tensor1 by tensor2, multiplies the result by the scalar value and adds it to input.
input_tensor = torch.tensor([1, 2, 3])
tensor1 = torch.tensor([4, 5, 6])
tensor2 = torch.tensor([2, 2, 2])
value = 0.5
addcmul_tensor = torch.addcmul(input_tensor, value, tensor1, tensor2)
print(addcmul_tensor)

# angle
# Computes the element-wise angle (in radians) of the given input tensor.
input_tensor = torch.complex(torch.tensor([1., 1., -1., -1.]), torch.tensor([0., 1., 0., -1.]))
angle_tensor = torch.angle(input_tensor)
print(angle_tensor)

# asin
# Returns a new tensor with the arcsine of the elements of input.
input_tensor = torch.tensor([0.5, -0.5])
asin_tensor = torch.asin(input_tensor)
print(asin_tensor)

# asinh
# Returns a new tensor with the inverse hyperbolic sine of the elements of input.
input_tensor = torch.tensor([1, 2, 3])
asinh_tensor = torch.asinh(input_tensor)
print(asinh_tensor)

# atan
# Returns a new tensor with the arctangent of the elements of input.
input_tensor = torch.tensor([1, 0, -1])
atan_tensor = torch.atan(input_tensor)
print(atan_tensor)

# atanh
# Returns a new tensor with the inverse hyperbolic tangent of the elements of input.
input_tensor = torch.tensor([0.5, 0, -0.5])
atanh_tensor = torch.atanh(input_tensor)
print(atanh_tensor)

# atan2
# Element-wise arctangent of input1 / input2 with consideration of the quadrant.
input_tensor1 = torch.tensor([1.0, -1.0])
input_tensor2 = torch.tensor([1.0, 1.0])
atan2_tensor = torch.atan2(input_tensor1, input_tensor2)
print(atan2_tensor)

# bitwise_not
# Computes the bitwise NOT of the given input tensor.
input_tensor = torch.tensor([1, 2, 3])
bitwise_not_tensor = torch.bitwise_not(input_tensor)
print(bitwise_not_tensor)

# bitwise_and
# Computes the bitwise AND of input and other.
input_tensor = torch.tensor([1, 2, 3])
other_tensor = torch.tensor([2, 3, 4])
bitwise_and_tensor = torch.bitwise_and(input_tensor, other_tensor)
print(bitwise_and_tensor)

# bitwise_or
# Computes the bitwise OR of input and other.
input_tensor = torch.tensor([1, 2, 3])
other_tensor = torch.tensor([2, 3, 4])
bitwise_or_tensor = torch.bitwise_or(input_tensor, other_tensor)
print(bitwise_or_tensor)

# bitwise_xor
# Computes the bitwise XOR of input and other.
input_tensor = torch.tensor([1, 2, 3])
other_tensor = torch.tensor([2, 3, 4])
bitwise_xor_tensor = torch.bitwise_xor(input_tensor, other_tensor)
print(bitwise_xor_tensor)

# bitwise_left_shift
# Computes the left arithmetic shift of input by other bits.
input_tensor = torch.tensor([1, 2, 3])
other_tensor = torch.tensor([1, 2, 3])
bitwise_left_shift_tensor = torch.bitwise_left_shift(input_tensor, other_tensor)
print(bitwise_left_shift_tensor)

# bitwise_right_shift
# Computes the right arithmetic shift of input by other bits.
input_tensor = torch.tensor([4, 8, 16])
other_tensor = torch.tensor([1, 2, 3])
bitwise_right_shift_tensor = torch.bitwise_right_shift(input_tensor, other_tensor)
print(bitwise_right_shift_tensor)

# ceil
# Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
input_tensor = torch.tensor([1.4, 2.6, -3.2])
ceil_tensor = torch.ceil(input_tensor)
print(ceil_tensor)

# clamp
# Clamps all elements in input into the range [min, max].
input_tensor = torch.tensor([-1, 2, 3])
clamped_tensor = torch.clamp(input_tensor, min=-1, max=2)
print(clamped_tensor)

# conj_physical
# Computes the element-wise conjugate of the given input tensor.
input_tensor = torch.complex(torch.tensor([1., 1., -1., -1.]), torch.tensor([0., 1., 0., -1.]))
conj_physical_tensor = torch.conj_physical(input_tensor)
print(conj_physical_tensor)

# copysign
# Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.
input_tensor = torch.tensor([-1.5, 2.5, -3.5])
other_tensor = torch.tensor([1.0, -1.0, 1.0])
copysign_tensor = torch.copysign(input_tensor, other_tensor)
print(copysign_tensor)

# cos
# Returns a new tensor with the cosine of the elements of input.
input_tensor = torch.tensor([0, 90, 180])
cos_tensor = torch.cos(torch.deg2rad(input_tensor))
print(cos_tensor)

# cosh
# Returns a new tensor with the hyperbolic cosine of the elements of input.
input_tensor = torch.tensor([0, 1, 2])
cosh_tensor = torch.cosh(input_tensor)
print(cosh_tensor)

# deg2rad
# Returns a new tensor with each of the elements of input converted from angles in degrees to radians.
input_tensor = torch.tensor([0, 90, 180])
rad_tensor = torch.deg2rad(input_tensor)
print(rad_tensor)

# div
# Divides each element of the input tensor by the corresponding element of other.
input_tensor = torch.tensor([4, 6, 8])
other_tensor = torch.tensor([2, 2, 2])
div_tensor = torch.div(input_tensor, other_tensor)
print(div_tensor)

# erf
# Computes the error function of each element in input.
input_tensor = torch.tensor([0.0, 0.5, 1.0])
erf_tensor = torch.erf(input_tensor)
print(erf_tensor)

# erfc
# Computes the complementary error function of each element in input.
input_tensor = torch.tensor([0.0, 0.5, 1.0])
erfc_tensor = torch.erfc(input_tensor)
print(erfc_tensor)

# exp
# Returns a new tensor with the exponential of the elements of the input tensor.
input_tensor = torch.tensor([1, 2, 3])
exp_tensor = torch.exp(input_tensor)
print(exp_tensor)

# exp2
# Returns a new tensor with the base 2 exponential of the elements of the input tensor.
input_tensor = torch.tensor([1, 2, 3])
exp2_tensor = torch.exp2(input_tensor)
print(exp2_tensor)

# expm1
# Returns a new tensor with the exponential of the elements minus 1 of the input tensor.
input_tensor = torch.tensor([1, 2, 3])
expm1_tensor = torch.expm1(input_tensor)
print(expm1_tensor)

# fake_quantize_per_channel_affine
# Returns a new tensor with the data in input fake quantized per channel using scale, zero_point, quant_min, and quant_max, across the channel specified by axis.
#input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#scale_tensor = torch.tensor([0.1, 0.2, 0.3])
#zero_point_tensor = torch.tensor([1, 2, 3])
#quant_min = 0
#quant_max = 255
#axis = 1
#fake_quantize_per_channel_affine_tensor = torch.fake_quantize_per_channel_affine(input_tensor, scale_tensor, zero_point_tensor, quant_min, quant_max, axis)
#print(fake_quantize_per_channel_affine_tensor)

# fake_quantize_per_tensor_affine
# Returns a new tensor with the data in input fake quantized using scale, zero_point, quant_min, and quant_max.
input_tensor = torch.tensor([1.0, 2.0, 3.0])
scale = 0.1
zero_point = 1
quant_min = 0
quant_max = 255
fake_quantize_per_tensor_affine_tensor = torch.fake_quantize_per_tensor_affine(input_tensor, scale, zero_point, quant_min, quant_max)
print(fake_quantize_per_tensor_affine_tensor)

