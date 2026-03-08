import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Objective function: F(x) = sum(x)
def objective(x):
    return np.sum(x, axis=1)

# Generate samples from N(0, I)
np.random.seed(42)
samples = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=100)
mean_all = np.mean(samples, axis=0)
fitness = objective(samples)

# Select top 30 individuals by fitness
top_indices = np.argsort(fitness)[-30:]
top_samples = samples[top_indices]
mean_top = np.mean(top_samples, axis=0)

# Compute new covariance matrix from top samples
cov_new = np.cov(top_samples, rowvar=False)

# Draw ellipse for covariance matrix
def draw_cov_ellipse(ax, mean, cov, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# Draw background contours
def plot_contours(ax):
    X, Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    Z = X + Y
    ax.contour(X, Y, Z, levels=10, colors='gray', linestyles='dashed', alpha=0.5, linewidths=0.5)

# ----- Subplot 1: Sampling -----
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(samples[:, 0], samples[:, 1], s=10, c='black', alpha=0.5)
draw_cov_ellipse(ax, mean_all, np.eye(2), edgecolor='black', linestyle='--', fill=False)
plot_contours(ax)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title("Sampling")
plt.tight_layout()
plt.savefig("sampling_plot.png", dpi=800)
plt.close()

# ----- Subplot 2: Estimation -----
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_samples[:, 0], top_samples[:, 1], c='black', s=10)
ax.scatter(*mean_all, color='black', marker='x')
for pt in top_samples:
    ax.arrow(mean_all[0], mean_all[1],
             pt[0] - mean_all[0], pt[1] - mean_all[1],
             head_width=0.0, head_length=0.0, fc='black', ec='black', linewidth=0.1)
draw_cov_ellipse(ax, mean_top, cov_new, edgecolor='black', fill=False)
plot_contours(ax)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title("Estimation")
plt.tight_layout()
plt.savefig("estimation_plot.png", dpi=800)
plt.close()

# ----- Subplot 3: Updated Covariance -----
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(*mean_top, color='black', marker='x')
draw_cov_ellipse(ax, mean_all, np.eye(2), edgecolor='black', linestyle='--', fill=False)
draw_cov_ellipse(ax, mean_top, cov_new, edgecolor='black', fill=False)
plot_contours(ax)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title("Updated Covariance")
plt.tight_layout()
plt.savefig("updated_covariance_plot.png", dpi=800)
plt.close()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse

# # Objective function: F(x) = sum(x)
# def objective(x):
#     return np.sum(x, axis=1)

# # Generate samples from N(0, I)
# np.random.seed(42)
# samples = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=100)
# mean_all = np.mean(samples, axis=0)
# fitness = objective(samples)

# # Select top 20 individuals by fitness
# top_indices = np.argsort(fitness)[-30:]
# top_samples = samples[top_indices]
# mean_top = np.mean(top_samples, axis=0)

# # Compute new covariance matrix from top samples
# cov_new = np.cov(top_samples, rowvar=False)

# # Utility: Draw ellipse for covariance matrix
# def draw_cov_ellipse(ax, mean, cov, **kwargs):
#     vals, vecs = np.linalg.eigh(cov)
#     angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
#     width, height = 2 * np.sqrt(vals)
#     ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
#     ax.add_patch(ellipse)

# # Create background contours
# def plot_contours(ax):
#     X, Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
#     Z = X + Y
#     ax.contour(X, Y, Z, levels=10, colors='gray', linestyles='dashed', alpha=0.5, linewidths=0.5)

# # Plotting
# fig, axs = plt.subplots(1, 3, figsize=(9, 3))

# # ----- Left plot -----
# axs[0].scatter(samples[:, 0], samples[:, 1], s=10, c='black', alpha=0.5)
# draw_cov_ellipse(axs[0], mean_all, np.eye(2), edgecolor='black', linestyle='--', fill=False)
# plot_contours(axs[0])
# axs[0].set_title("Sampling")
# axs[0].set_xticks([])
# axs[0].set_yticks([])

# # ----- Middle plot -----
# axs[1].scatter(top_samples[:, 0], top_samples[:, 1], c='black', s=10)
# axs[1].scatter(*mean_all, color='black', marker='x')
# for pt in top_samples:
#     axs[1].arrow(mean_all[0], mean_all[1],
#                  pt[0] - mean_all[0], pt[1] - mean_all[1],
#                  head_width=0.0, head_length=0.0, fc='black', ec='black', linewidth=0.1)
# draw_cov_ellipse(axs[1], mean_top, cov_new, edgecolor='black', fill=False)
# plot_contours(axs[1])
# axs[1].set_title("Estimation")
# axs[1].set_xticks([])
# axs[1].set_yticks([])

# # ----- Right plot -----
# axs[2].scatter(*mean_top, color='black', marker='x')
# draw_cov_ellipse(axs[2], mean_all, np.eye(2), edgecolor='black', linestyle='--', fill=False)
# draw_cov_ellipse(axs[2], mean_top, cov_new, edgecolor='black', fill=False)
# plot_contours(axs[2])
# axs[2].set_title("Updated Covariance")
# axs[2].set_xticks([])
# axs[2].set_yticks([])

# # General style
# for ax in axs:
#     ax.set_xlim(-4, 4)
#     ax.set_ylim(-4, 4)
#     ax.set_aspect('equal')
#     ax.grid(False)

# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Time array
# t = np.linspace(0, 1.5, 1000)

# # Parameters
# prupt = 1.0
# delta_p = 0.1 * prupt
# t_rupture = 0.5
# t_rise_center = t_rupture + 0.03
# t_fall_start = t_rupture + 0.12

# # Smooth rise using sigmoid-like function
# rise = delta_p / (1 + np.exp(-200 * (t - t_rise_center)))

# # Smooth fall using exponential decay
# fall = delta_p * np.exp(-40 * (t - t_fall_start))
# fall[t < t_fall_start] = delta_p  # flat before decay begins

# # Combined pressure trace
# p = np.zeros_like(t)
# p[t < t_rupture] = 0
# p[(t >= t_rupture) & (t < t_fall_start)] = prupt + rise[(t >= t_rupture) & (t < t_fall_start)]
# p[t >= t_fall_start] = prupt + fall[t >= t_fall_start]

# # Plot
# fig, ax = plt.subplots(dpi=800)
# ax.plot(t, p, color='black', linewidth=1.5)

# # Reference lines
# ax.axhline(prupt, color='gray', linestyle=':', linewidth=0.8)
# ax.axhline(prupt + delta_p, color='gray', linestyle=':', linewidth=0.8)
# ax.axhline(prupt - delta_p, color='gray', linestyle=':', linewidth=0.8)
# ax.axvline(t_rupture, color='gray', linestyle=':', linewidth=0.8)

# # Annotations
# t_end = t_fall_start + 0.1
# t_start = t_rupture
# ax.annotate(
#     'useful supply time',
#     xy=((t_start + t_end) / 2, prupt + delta_p * 0.9),
#     xytext=(t_end + 0.05, prupt + 0.25),
#     arrowprops=dict(arrowstyle='-[,widthB=4.5,lengthB=0.75', lw=1),
#     fontsize=10
# )

# # Δp label
# ax.annotate(r'$\pm \Delta p$', xy=(t_end + 0.05, prupt), fontsize=12)
# ax.text(t_end + 0.02, prupt + 0.13, r'$\left(\frac{\Delta p}{p_{rupt}} \approx 10\%\right)$', fontsize=12)

# # Rupture point
# ax.plot(t_rupture, prupt, 'ko')
# ax.annotate("rupture", xy=(t_rupture - 0.05, prupt + 0.05), fontsize=10)

# # Axes labels
# ax.set_xlabel(r'$t$', fontsize=14)
# ax.set_ylabel(r'$p$', fontsize=14)
# ax.set_xlim(0, 1.3)
# ax.set_ylim(0, prupt + delta_p + 0.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.tight_layout()
# plt.show()
