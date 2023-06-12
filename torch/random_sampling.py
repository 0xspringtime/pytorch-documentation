import torch

# seed
# Sets the seed for generating random numbers to a non-deterministic random number.
torch.manual_seed(torch.initial_seed())

# manual_seed
# Sets the seed for generating random numbers.
torch.manual_seed(123)

# initial_seed
# Returns the initial seed for generating random numbers as a Python long.
initial_seed = torch.initial_seed()
print(initial_seed)

# get_rng_state
# Returns the random number generator state as a torch.ByteTensor.
rng_state = torch.get_rng_state()
print(rng_state)

# set_rng_state
# Sets the random number generator state.
#new_rng_state = torch.ByteTensor(rng_state.size())
#torch.set_rng_state(new_rng_state)

# bernoulli
# Draws binary random numbers (0 or 1) from a Bernoulli distribution.
probs = torch.tensor([0.3, 0.7])
bernoulli_tensor = torch.bernoulli(probs)
print(bernoulli_tensor)

# multinomial
# Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.
#weights = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
#num_samples = 5
#multinomial_tensor = torch.multinomial(weights, num_samples)
#print(multinomial_tensor)

# normal
# Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
mean = torch.tensor([0.0, 1.0])
std = torch.tensor([1.0, 2.0])
normal_tensor = torch.normal(mean, std)
print(normal_tensor)

# poisson
# Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input.
input_tensor = torch.tensor([1.0, 2.0, 3.0])
poisson_tensor = torch.poisson(input_tensor)
print(poisson_tensor)

# rand
# Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
rand_tensor = torch.rand(2, 3)
print(rand_tensor)

# rand_like
# Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0, 1)
#input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
#rand_like_tensor = torch.rand_like(input_tensor)
#print(rand_like_tensor)

# randint
# Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
low = 0
high = 10
randint_tensor = torch.randint(low, high, size=(2, 3))
print(randint_tensor)

# randint_like
# Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive).
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
randint_like_tensor = torch.randint_like(input_tensor, low, high)
print(randint_like_tensor)

# randn
# Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (standard normal distribution).
randn_tensor = torch.randn(2, 3)
print(randn_tensor)

# randn_like
# Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
#input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
#randn_like_tensor = torch.randn_like(input_tensor)
#print(randn_like_tensor)

# randperm
# Returns a random permutation of integers from 0 to n - 1.
n = 5
randperm_tensor = torch.randperm(n)
print(randperm_tensor)

