# Add these imports at the top if not already present
from collections import defaultdict

# Add these variables to the start of the training loop
kl_divergence = defaultdict(list)

# Inside the training loop, after calculating approx_kl, add the following:
if all(policy == "internal" for policy in used_policies):
    kl_divergence["internal"].append(approx_kl.item())
elif all(policy == "octo" for policy in used_policies):
    kl_divergence["octo"].append(approx_kl.item())

# After the training loop, log the KL divergence for both "octo" and "internal" actions
if self.logger is not None:
    if kl_divergence["internal"]:
        avg_kl_internal = np.mean(kl_divergence["internal"])
        self.logger.log_metrics("losses/avg_kl_internal", avg_kl_internal, global_step)
    if kl_divergence["octo"]:
        avg_kl_octo = np.mean(kl_divergence["octo"])
        self.logger.log_metrics("losses/avg_kl_octo", avg_kl_octo, global_step)
