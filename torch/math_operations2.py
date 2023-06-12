import torch

# float_power
# Raises input to the power of exponent, elementwise, in double precision.
input_tensor = torch.tensor([2.0, 3.0, 4.0])
exponent_tensor = torch.tensor([2.0, 2.0, 2.0])
float_power_tensor = torch.float_power(input_tensor, exponent_tensor)
print(float_power_tensor)

# floor
# Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
input_tensor = torch.tensor([1.4, 2.6, 3.2])
floor_tensor = torch.floor(input_tensor)
print(floor_tensor)

# floor_divide
# Returns the largest integer less than or equal to the division of the inputs.
input_tensor1 = torch.tensor([5, 7, 9])
input_tensor2 = torch.tensor([2, 3, 4])
floor_divide_tensor = torch.floor_divide(input_tensor1, input_tensor2)
print(floor_divide_tensor)

# fmod
# Applies C++'s std::fmod entrywise.
input_tensor1 = torch.tensor([5.5, 7.5, 9.5])
input_tensor2 = torch.tensor([2.2, 3.3, 4.4])
fmod_tensor = torch.fmod(input_tensor1, input_tensor2)
print(fmod_tensor)

# frac
# Computes the fractional portion of each element in input.
input_tensor = torch.tensor([1.5, 2.7, -3.4])
frac_tensor = torch.frac(input_tensor)
print(frac_tensor)

# frexp
# Decomposes input into mantissa and exponent tensors such that input = mantissa × 2 ** exponent.
input_tensor = torch.tensor([2.0, 3.0, 4.0])
mantissa_tensor, exponent_tensor = torch.frexp(input_tensor)
print(mantissa_tensor)
print(exponent_tensor)

# gradient
# Estimates the gradient of a function f: R^n -> R in one or more dimensions using the second-order accurate central differences method.
#def function(x):
#    return x ** 2
#
#input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
#gradient_tensor = torch.gradient(function(input_tensor), input_tensor)
#print(gradient_tensor)

# imag
# Returns a new tensor containing imaginary values of the self tensor.
input_tensor = torch.tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype=torch.complex128)
imag_tensor = torch.imag(input_tensor)
print(imag_tensor)

# ldexp
# Multiplies input by 2 ** other.
input_tensor = torch.tensor([1.0, 2.0, 3.0])
other_tensor = torch.tensor([2, 3, 4])
ldexp_tensor = torch.ldexp(input_tensor, other_tensor)
print(ldexp_tensor)

# lerp
# Does a linear interpolation of two tensors start (given by input) and end based on a scalar or tensor weight and returns the resulting out tensor.
start_tensor = torch.tensor([1.0, 2.0, 3.0])
end_tensor = torch.tensor([4.0, 5.0, 6.0])
weight = 0.5
lerp_tensor = torch.lerp(start_tensor, end_tensor, weight)
print(lerp_tensor)

# lgamma
# Computes the natural logarithm of the absolute value of the gamma function on input.
input_tensor = torch.tensor([1.0, 2.0, 3.0])
lgamma_tensor = torch.lgamma(input_tensor)
print(lgamma_tensor)

# log
# Returns a new tensor with the natural logarithm of the elements of input.
input_tensor = torch.tensor([1.0, 2.0, 3.0])
log_tensor = torch.log(input_tensor)
print(log_tensor)

# log10
# Returns a new tensor with the logarithm to the base 10 of the elements of input.
input_tensor = torch.tensor([1.0, 10.0, 100.0])
log10_tensor = torch.log10(input_tensor)
print(log10_tensor)

# log1p
# Returns a new tensor with the natural logarithm of (1 + input).
input_tensor = torch.tensor([0.0, 0.5, 1.0])
log1p_tensor = torch.log1p(input_tensor)
print(log1p_tensor)

# log2
# Returns a new tensor with the logarithm to the base 2 of the elements of input.
input_tensor = torch.tensor([1.0, 2.0, 4.0])
log2_tensor = torch.log2(input_tensor)
print(log2_tensor)

# logaddexp
# Logarithm of the sum of exponentiations of the inputs.
input_tensor1 = torch.tensor([1.0, 2.0, 3.0])
input_tensor2 = torch.tensor([4.0, 5.0, 6.0])
logaddexp_tensor = torch.logaddexp(input_tensor1, input_tensor2)
print(logaddexp_tensor)

# logaddexp2
# Logarithm of the sum of exponentiations of the inputs in base-2.
input_tensor1 = torch.tensor([1.0, 2.0, 3.0])
input_tensor2 = torch.tensor([4.0, 5.0, 6.0])
logaddexp2_tensor = torch.logaddexp2(input_tensor1, input_tensor2)
print(logaddexp2_tensor)

# logical_and
# Computes the element-wise logical AND of the given input tensors.
input_tensor1 = torch.tensor([True, False, True])
input_tensor2 = torch.tensor([True, True, False])
logical_and_tensor = torch.logical_and(input_tensor1, input_tensor2)
print(logical_and_tensor)

# logical_not
# Computes the element-wise logical NOT of the given input tensor.
input_tensor = torch.tensor([True, False, True])
logical_not_tensor = torch.logical_not(input_tensor)
print(logical_not_tensor)

# logical_or
# Computes the element-wise logical OR of the given input tensors.
input_tensor1 = torch.tensor([True, False, True])
input_tensor2 = torch.tensor([True, True, False])
logical_or_tensor = torch.logical_or(input_tensor1, input_tensor2)
print(logical_or_tensor)

# logical_xor
# Computes the element-wise logical XOR of the given input tensors.
input_tensor1 = torch.tensor([True, False, True])
input_tensor2 = torch.tensor([True, True, False])
logical_xor_tensor = torch.logical_xor(input_tensor1, input_tensor2)
print(logical_xor_tensor)

# logit
# Computes the logit function, the inverse of the sigmoid function.
input_tensor = torch.tensor([0.25, 0.5, 0.75])
logit_tensor = torch.logit(input_tensor)
print(logit_tensor)

# hypot
# Given the legs of a right triangle, return its hypotenuse.
input_tensor1 = torch.tensor([3.0, 4.0, 5.0])
input_tensor2 = torch.tensor([4.0, 5.0, 6.0])
hypot_tensor = torch.hypot(input_tensor1, input_tensor2)
print(hypot_tensor)

# i0
# Computes the zeroth order modified Bessel function of the first kind of input.
input_tensor = torch.tensor([1.0, 2.0, 3.0])
i0_tensor = torch.i0(input_tensor)
print(i0_tensor)

# igamma
# Computes the regularized lower incomplete gamma function of input.
input_tensor1 = torch.tensor([1.0, 2.0, 3.0])
input_tensor2 = torch.tensor([2.0, 3.0, 4.0])
igamma_tensor = torch.igamma(input_tensor1, input_tensor2)
print(igamma_tensor)

# igammac
# Computes the regularized upper incomplete gamma function of input.
input_tensor1 = torch.tensor([1.0, 2.0, 3.0])
input_tensor2 = torch.tensor([2.0, 3.0, 4.0])
igammac_tensor = torch.igammac(input_tensor1, input_tensor2)
print(igammac_tensor)

# mul
# Multiplies input by other.
input_tensor1 = torch.tensor([2, 3, 4])
input_tensor2 = torch.tensor([5, 6, 7])
mul_tensor = torch.mul(input_tensor1, input_tensor2)
print(mul_tensor)

# multiply
# Alias for torch.mul().
input_tensor1 = torch.tensor([2, 3, 4])
input_tensor2 = torch.tensor([5, 6, 7])
multiply_tensor = torch.multiply(input_tensor1, input_tensor2)
print(multiply_tensor)

# mvlgamma
# Computes the multivariate log-gamma function.
input_tensor = torch.tensor([1.0, 2.0, 3.0])
mvlgamma_tensor = torch.mvlgamma(input_tensor, 3)
print(mvlgamma_tensor)

# nan_to_num
# Replaces NaN, positive infinity, and negative infinity values in input with the values specified.
input_tensor = torch.tensor([1.0, float('nan'), float('inf'), float('-inf')])
nan_to_num_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
print(nan_to_num_tensor)

# neg
# Returns a new tensor with the negative of the elements of input.
input_tensor = torch.tensor([1, -2, 3])
neg_tensor = torch.neg(input_tensor)
print(neg_tensor)

# negative
# Alias for torch.neg().
input_tensor = torch.tensor([1, -2, 3])
negative_tensor = torch.negative(input_tensor)
print(negative_tensor)

# nextafter
# Return the next floating-point value after input towards other, elementwise.
input_tensor1 = torch.tensor([1.0, 2.0, 3.0])
input_tensor2 = torch.tensor([2.0, 3.0, 4.0])
nextafter_tensor = torch.nextafter(input_tensor1, input_tensor2)
print(nextafter_tensor)

# polygamma
# Computes the nth derivative of the digamma function (also called psi).
#input_tensor1 = torch.tensor([1.0, 2.0, 3.0])
#input_tensor2 = torch.tensor([2.0, 3.0, 4.0])
#polygamma_tensor = torch.polygamma(input_tensor1, input_tensor2)
#print(polygamma_tensor)

# positive
# Returns input.
input_tensor = torch.tensor([1, -2, 3])
positive_tensor = torch.positive(input_tensor)
print(positive_tensor)

# pow
# Takes the power of each element in input with exponent and returns a tensor with the result.
input_tensor = torch.tensor([2.0, 3.0, 4.0])
exponent_tensor = torch.tensor([2.0, 2.0, 2.0])
pow_tensor = torch.pow(input_tensor, exponent_tensor)
print(pow_tensor)

# quantized_batch_norm
# Applies batch normalization on a quantized tensor.
#input_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.quint8)
#scale_tensor = torch.tensor([1.0])
#zero_point_tensor = torch.tensor([0])
#quantized_batch_norm_tensor = torch.quantized_batch_norm(input_tensor, scale_tensor, zero_point_tensor)
#print(quantized_batch_norm_tensor)

# quantized_max_pool1d
# Applies a 1D max pooling over a quantized tensor.
#input_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.quint8)
#quantized_max_pool1d_tensor = torch.quantized_max_pool1d(input_tensor, kernel_size=2)
#print(quantized_max_pool1d_tensor)

# quantized_max_pool2d
# Applies a 2D max pooling over a quantized tensor.
#input_tensor = torch.tensor([[0, 1], [2, 3]], dtype=torch.quint8)
#quantized_max_pool2d_tensor = torch.quantized_max_pool2d(input_tensor, kernel_size=2)
#print(quantized_max_pool2d_tensor)

# rad2deg
# Returns a new tensor with each of the elements of input converted from angles in radians to degrees.
input_tensor = torch.tensor([0.0, 3.14159, 6.28318])
rad2deg_tensor = torch.rad2deg(input_tensor)
print(rad2deg_tensor)

# real
# Returns a new tensor containing real values of the self tensor.
input_tensor = torch.tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype=torch.complex128)
real_tensor = torch.real(input_tensor)
print(real_tensor)

# reciprocal
# Returns a new tensor with the reciprocal of the elements of input.
input_tensor = torch.tensor([2.0, 0.5, -0.25])
reciprocal_tensor = torch.reciprocal(input_tensor)
print(reciprocal_tensor)

# remainder
# Computes the remainder of division elementwise.
input_tensor1 = torch.tensor([5.0, 7.0, 9.0])
input_tensor2 = torch.tensor([2.0, 3.0, 4.0])
remainder_tensor = torch.remainder(input_tensor1, input_tensor2)
print(remainder_tensor)

# round
# Rounds elements of input to the nearest integer.
input_tensor = torch.tensor([1.4, 2.6, 3.5])
round_tensor = torch.round(input_tensor)
print(round_tensor)

# rsqrt
# Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
input_tensor = torch.tensor([4.0, 9.0, 16.0])
rsqrt_tensor = torch.rsqrt(input_tensor)
print(rsqrt_tensor)

# sigmoid
# Applies the element-wise sigmoid function.
input_tensor = torch.tensor([0.0, 1.0, 2.0])
sigmoid_tensor = torch.sigmoid(input_tensor)
print(sigmoid_tensor)

# sign
# Returns a new tensor with the signs of the elements of input.
input_tensor = torch.tensor([1.0, -2.0, 3.0])
sign_tensor = torch.sign(input_tensor)
print(sign_tensor)

# sgn
# This function is an extension of torch.sign() to complex tensors.
input_tensor = torch.tensor([1 + 2j, -3 + 4j, 5 - 6j], dtype=torch.complex128)
sgn_tensor = torch.sgn(input_tensor)
print(sgn_tensor)

# signbit
# Tests if each element of input has its sign bit set or not.
input_tensor = torch.tensor([-1.0, 2.0, -3.0])
signbit_tensor = torch.signbit(input_tensor)
print(signbit_tensor)

# sin
# Returns a new tensor with the sine of the elements of input.
input_tensor = torch.tensor([0.0, 1.0, 2.0])
sin_tensor = torch.sin(input_tensor)
print(sin_tensor)

# sinc
# Computes the sinc function of input.
input_tensor = torch.tensor([0.0, 1.0, 2.0])
sinc_tensor = torch.sinc(input_tensor)
print(sinc_tensor)

# sinh
# Returns a new tensor with the hyperbolic sine of the elements of input.
input_tensor = torch.tensor([0.0, 1.0, 2.0])
sinh_tensor = torch.sinh(input_tensor)
print(sinh_tensor)

# softmax
# Applies the Softmax function to an input Tensor.
input_tensor = torch.tensor([1.0, 2.0, 3.0])
softmax_tensor = torch.softmax(input_tensor, dim=0)
print(softmax_tensor)

# sqrt
# Returns a new tensor with the square root of the elements of input.
input_tensor = torch.tensor([4.0, 9.0, 16.0])
sqrt_tensor = torch.sqrt(input_tensor)
print(sqrt_tensor)

# square
# Returns a new tensor with the square of the elements of input.
input_tensor = torch.tensor([2.0, 3.0, 4.0])
square_tensor = torch.square(input_tensor)
print(square_tensor)

# sub
# Subtracts other, scaled by alpha, from input.
input_tensor1 = torch.tensor([2.0, 3.0, 4.0])
input_tensor2 = torch.tensor([1.0, 2.0, 3.0])
sub_tensor = torch.sub(input_tensor1, input_tensor2)
print(sub_tensor)

# subtract
# Alias for torch.sub().
input_tensor1 = torch.tensor([2.0, 3.0, 4.0])
input_tensor2 = torch.tensor([1.0, 2.0, 3.0])
subtract_tensor = torch.subtract(input_tensor1, input_tensor2)
print(subtract_tensor)

# tan
# Returns a new tensor with the tangent of the elements of input.
input_tensor = torch.tensor([0.0, 1.0, 2.0])
tan_tensor = torch.tan(input_tensor)
print(tan_tensor)

# tanh
# Returns a new tensor with the hyperbolic tangent of the elements of input.
input_tensor = torch.tensor([0.0, 1.0, 2.0])
tanh_tensor = torch.tanh(input_tensor)
print(tanh_tensor)

# true_divide
# Alias for torch.div() with rounding_mode=None.
input_tensor1 = torch.tensor([5.0, 7.0, 9.0])
input_tensor2 = torch.tensor([2.0, 3.0, 4.0])
true_divide_tensor = torch.true_divide(input_tensor1, input_tensor2)
print(true_divide_tensor)

# trunc
# Returns a new tensor with the truncated integer values of the elements of input.
input_tensor = torch.tensor([1.4, 2.6, 3.9])
trunc_tensor = torch.trunc(input_tensor)
print(trunc_tensor)

# xlogy
# Computes the element-wise x * log(y) of the input tensors.
input_tensor1 = torch.tensor([1.0, 2.0, 3.0])
input_tensor2 = torch.tensor([2.0, 3.0, 4.0])
xlogy_tensor = torch.xlogy(input_tensor1, input_tensor2)
print(xlogy_tensor)

