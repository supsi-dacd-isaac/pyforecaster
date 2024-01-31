import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfinv

def lognorm_crps(sigma_square, y_hat, y, distribution='normal'):
    if distribution == 'normal':
        sigma = jnp.sqrt(sigma_square)
        mu = y_hat
        u = (y - mu) / sigma
        crps = sigma * (u * (2 * jax.scipy.stats.norm.cdf(u) - 1) +
                        2 * jax.scipy.stats.norm.pdf(u) - 1 / jnp.sqrt(jnp.pi))
        return crps
    elif distribution == 'log-normal':
        s = jnp.sign(sigma_square)
        sigma = jnp.sqrt(jnp.abs(sigma_square)) + 1e-10

        # we have predicted the mean of the log-normal distribution, we need to recover the parameter mu
        mu = jnp.log(y_hat ** 2 / jnp.sqrt(jnp.abs(sigma_square) + y_hat ** 2))
        sigma_square = jnp.log(1 + sigma ** 2 / y_hat ** 2)
        sigma = jnp.sqrt(sigma_square)

        # Standardize the observed value
        z = (jnp.log(jnp.maximum(1e-10, s * y + 2 * jnp.exp(mu) * (s == -1))) - mu) / sigma

        cdf_2 = jax.scipy.stats.norm.cdf(sigma / jnp.sqrt(2))
        cdf_zs = (s == -1) + s * jax.scipy.stats.norm.cdf(z - sigma)
        cdf_z = (s == -1) + s * jax.scipy.stats.norm.cdf(z)

        crps = y * (2 * cdf_z - 1) - 2 * jnp.exp(mu + np.abs(sigma_square) / 2) * (
                cdf_zs + cdf_2 - 1)

        return cdf_z, cdf_zs, crps

def quantiles(sigma_square, y_hat, q_vect, distribution='normal'):
    s = np.sign(sigma_square)
    sigma_hat = np.abs(sigma_square)**0.5
    preds = np.expand_dims(y_hat, -1) * np.ones((1, 1, len(q_vect)))
    for i, q in enumerate(q_vect):
        if distribution == 'normal':
            qs = sigma_hat * np.sqrt(2) * erfinv(2 * q - 1)
            preds[:, :, i] += qs
        elif distribution == 'log-normal':
            # the transformation to flip the quantile function in the case of negative sigma_square is the following:
            # -F(1-alpha) + 2*exp(mu)
            # where F is the original quantile function
            sg_hat_square = jnp.log(1 + sigma_hat ** 2 / y_hat ** 2)
            sg_hat = jnp.sqrt(sg_hat_square)
            mu_hat = jnp.log(y_hat ** 2 / jnp.sqrt(sigma_hat ** 2 + y_hat ** 2))
            qp = mu_hat + sg_hat * np.sqrt(2) * erfinv(2 * q - 1)
            pos_qs = np.exp(qp)
            qn = mu_hat + sg_hat * np.sqrt(2) * erfinv(2 * (1 - q) - 1)
            neg_qs = -np.exp(qn) + 2*np.exp(mu_hat)
            preds[:, :, i] = (s==-1)*neg_qs + (s==1)*pos_qs
    return preds

sigma_square = 0.9**2
y_hat = 1
y = np.linspace(0, y_hat*2, 100)
crps = lognorm_crps(sigma_square, y_hat, y, distribution='normal')

fig, ax= plt.subplots(1, 1)
ax.plot(y, crps, label='cdf_z')
ax.vlines(y_hat, 0, 1, label='y_hat', color='violet')
ax.legend()


sigma_square = -2**2
y_hat = 1
y = np.linspace(-y_hat, y_hat*2, 100)
cdf_z, cdf_zs, crps = lognorm_crps(sigma_square, y_hat, y, distribution='log-normal')
alphas = np.linspace(0.1, 0.9, 100)
qs = quantiles(sigma_square, y_hat, alphas, distribution='log-normal')

fig, ax= plt.subplots(2, 1, figsize=(10, 10),sharex=True)
ax[0].plot(y, cdf_z, label='cdf_z')
ax[0].plot(y, cdf_zs, label='cdf_zs')
ax[0].plot(y, crps, label='crps')
ax[0].vlines(y_hat, 0, 1, label='y_hat', color='violet')
ax[0].plot(qs.ravel(), alphas, label='q=0', marker='.', linestyle='None')

mu = jnp.log(y_hat ** 2 / jnp.sqrt(jnp.abs(sigma_square) + y_hat ** 2))
ax[0].vlines(np.exp(mu), 0, 1, label='exp(mu)', color='violet', linestyle='--')
ax[0].legend()
ax[1].plot(y[:-1], np.diff(cdf_z), label='pdf_z')


sigma_square = 2**2
y_hat = 1
y = np.linspace(0, y_hat*2, 100)
cdf_z, cdf_zs, crps = lognorm_crps(sigma_square, y_hat, y, distribution='log-normal')
qs = quantiles(sigma_square, y_hat, alphas, distribution='log-normal')

fig, ax= plt.subplots(2, 1, figsize=(10, 10),sharex=True)
ax[0].plot(y, cdf_z, label='cdf_z')
ax[0].plot(y, cdf_zs, label='cdf_zs')
ax[0].plot(y, crps, label='crps')
ax[0].vlines(y_hat, 0, 1, label='y_hat', color='violet')
ax[0].plot(qs.ravel(), alphas, label='q=0', marker='.', linestyle='None')

mu = jnp.log(y_hat ** 2 / jnp.sqrt(jnp.abs(sigma_square) + y_hat ** 2))
ax[0].vlines(np.exp(mu), 0, 1, label='exp(mu)', color='violet', linestyle='--')
ax[0].legend()
ax[1].plot(y[:-1], np.diff(cdf_z), label='pdf_z')
