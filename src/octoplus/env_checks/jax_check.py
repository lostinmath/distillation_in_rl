from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)

# jax with cuda can be installed by:
#  pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Beware: Torch must be cpu version than


"""
# checking jax has cuda support:
# installing with cmd gave cuda support for 12, tested on 12.5
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

sudo apt update && sudo apt upgrade
print(xla_bridge.get_backend().platform)
"""
