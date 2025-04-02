from vmas.simulator.core import make_env

# Create the Simple Tag environment
env = make_env('simple_tag')

# Get the observation spec
observation_spec = env.observation_space

# Print it
print(observation_spec)