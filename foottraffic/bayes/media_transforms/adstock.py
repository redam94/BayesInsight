"""
Transforms from PYMC marketing 
https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py
"""
import pytensor.tensor as pt
import pymc as pm
import jax.numpy as jnp
import jax
from enum import Enum, StrEnum
from pytensor.tensor.random.utils import params_broadcast_shapes

class ConvMode(StrEnum):
    After = "After"
    Before = "Before"
    Overlap = "Overlap"

class WeibullType(str, Enum):
    # TODO: use StrEnum when we upgrade to python 3.11
    PDF = "PDF"
    CDF = "CDF"


def batched_convolution(
    x,
    w,
    axis: int = 0,
    mode: ConvMode | str = ConvMode.After,
):
    R"""Apply a 1D convolution in a vectorized way across multiple batch dimensions.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import batched_convolution, ConvMode
        plt.style.use('arviz-darkgrid')
        spends = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        w = np.array([0.75, 0.25, 0.125, 0.125])
        x = np.arange(-5, 6)
        ax = plt.subplot(111)
        for mode in [ConvMode.Before, ConvMode.Overlap, ConvMode.After]:
            y = batched_convolution(spends, w, mode=mode).eval()
            suffix = "\n(default)" if mode == ConvMode.Before else ""
            plt.plot(x, y, label=f'{mode.value}{suffix}')
        plt.xlabel('time since spend', fontsize=12)
        plt.ylabel('f(time since spend)', fontsize=12)
        plt.title(f"1 spend at time 0 and {w = }", fontsize=14)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    Parameters
    ----------
    x :
        The array to convolve.
    w :
        The weight of the convolution. The last axis of ``w`` determines the number of steps
        to use in the convolution.
    axis : int
        The axis of ``x`` along witch to apply the convolution
    mode : ConvMode, optional
        The convolution mode determines how the convolution is applied at the boundaries
        of the input signal, denoted as "x." The default mode is ConvMode.Before.

        - ConvMode.After: Applies the convolution with the "Adstock" effect, resulting in a trailing decay effect.
        - ConvMode.Before: Applies the convolution with the "Excitement" effect, creating a leading effect
            similar to the wow factor.
        - ConvMode.Overlap: Applies the convolution with both "Pull-Forward" and "Pull-Backward" effects,
            where the effect overlaps with both preceding and succeeding elements.

    Returns
    -------
    y :
        The result of convolving ``x`` with ``w`` along the desired axis. The shape of the
        result will match the shape of ``x`` up to broadcasting with ``w``. The convolved
        axis will show the results of left padding zeros to ``x`` while applying the
        convolutions.
    """
    # We move the axis to the last dimension of the array so that it's easier to
    # reason about parameter broadcasting. We will move the axis back at the end
    orig_ndim = x.ndim
    axis = axis if axis >= 0 else orig_ndim + axis
    w = pt.as_tensor(w)
    x = pt.moveaxis(x, axis, -1)
    l_max = w.type.shape[-1]
    if l_max is None:
        try:
            l_max = w.shape[-1].eval()
        except Exception:  # noqa: S110
            pass
    # Get the broadcast shapes of x and w but ignoring their last dimension.
    # The last dimension of x is the "time" axis, which doesn't get broadcast
    # The last dimension of w is the number of time steps that go into the convolution
    x_shape, w_shape = params_broadcast_shapes([x.shape, w.shape], [1, 1])

    x = pt.broadcast_to(x, x_shape)
    w = pt.broadcast_to(w, w_shape)
    x_time = x.shape[-1]
    # Make a tensor with x at the different time lags needed for the convolution
    x_shape = x.shape
    # Add the size of the kernel to the time axis
    shape = (*x_shape[:-1], x_shape[-1] + w.shape[-1] - 1, w.shape[-1])
    padded_x = pt.zeros(shape, dtype=x.dtype)

    if l_max is None:  # pragma: no cover
        raise NotImplementedError(
            "At the moment, convolving with weight arrays that don't have a concrete shape "
            "at compile time is not supported."
        )
    # The window is the slice of the padded array that corresponds to the original x
    if l_max <= 1:
        window = slice(None)
    elif mode == ConvMode.Before:
        window = slice(l_max - 1, None)
    elif mode == ConvMode.After:
        window = slice(None, -l_max + 1)
    elif mode == ConvMode.Overlap:
        # Handle even and odd l_max differently if l_max is odd then we can split evenly otherwise we drop from the end
        window = slice((l_max // 2) - (1 if l_max % 2 == 0 else 0), -(l_max // 2))
    else:
        raise ValueError(f"Wrong Mode: {mode}, expected of ConvMode")

    for i in range(l_max):
        padded_x = pt.set_subtensor(padded_x[..., i : x_time + i, i], x)

    padded_x = padded_x[..., window, :]

    # The convolution is treated as an element-wise product, that then gets reduced
    # along the dimension that represents the convolution time lags
    conv = pt.sum(padded_x * w[..., None, :], axis=-1)
    # Move the "time" axis back to where it was in the original x array
    return pt.moveaxis(conv, -1, axis + conv.ndim - orig_ndim)
  
def delayed_adstock(
    x,
    alpha: float = 0.0,
    theta: int = 0,
    l_max: int = 12,
    normalize: bool = False,
    axis: int = 0,
    mode: ConvMode = ConvMode.After,
):
    R"""Delayed adstock transformation.

    This transformation is similar to geometric adstock transformation, but it
    allows for a delayed peak of the effect. The peak is assumed to occur at `theta`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import delayed_adstock
        plt.style.use('arviz-darkgrid')
        params = [
            (0.25, 0, False),
            (0.25, 5, False),
            (0.75, 5, False),
            (0.75, 5, True)
        ]
        spend = np.zeros(15)
        spend[0] = 1
        x = np.arange(len(spend))
        ax = plt.subplot(111)
        for a, t, normalize in params:
            y = delayed_adstock(spend, alpha=a, theta=t, normalize=normalize).eval()
            plt.plot(x, y, label=f'alpha = {a}\ntheta = {t}\nnormalize = {normalize}')
        plt.xlabel('time since spend', fontsize=12)
        plt.ylabel('f(time since spend)', fontsize=12)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    Parameters
    ----------
    x : tensor
        Input tensor.
    alpha : float, by default 0.0
        Retention rate of ad effect. Must be between 0 and 1.
    theta : float, by default 0
        Delay of the peak effect. Must be between 0 and `l_max` - 1.
    l_max : int, by default 12
        Maximum duration of carryover effect.
    normalize : bool, by default False
        Whether to normalize the weights.
    axis : int
        The axis of ``x`` along witch to apply the convolution
    mode : ConvMode, optional
        The convolution mode determines how the convolution is applied at the boundaries
        of the input signal, denoted as "x." The default mode is ConvMode.Before.

        - ConvMode.After: Applies the convolution with the "Adstock" effect, resulting in a trailing decay effect.
        - ConvMode.Before: Applies the convolution with the "Excitement" effect, creating a leading effect
            similar to the wow factor.
        - ConvMode.Overlap: Applies the convolution with both "Pull-Forward" and "Pull-Backward" effects,
            where the effect overlaps with both preceding and succeeding elements.

    Returns
    -------
    tensor
        Transformed tensor.

    References
    ----------
    .. [1] Jin, Yuxue, et al. "Bayesian methods for media mix modeling
       with carryover and shape effects." (2017).
    """
    w = pt.power(
        pt.as_tensor(alpha)[..., None],
        (pt.arange(l_max, dtype=x.dtype) - pt.as_tensor(theta)[..., None]) ** 2,
    )
    w = w / pt.sum(w, axis=-1, keepdims=True) if normalize else w
    return batched_convolution(x, w, axis=axis, mode=mode)
  
def weibull_adstock(
    x,
    lam=1,
    k=1,
    l_max: int = 12,
    axis: int = 0,
    mode: ConvMode = ConvMode.After,
    type: WeibullType | str = WeibullType.PDF,
):
    R"""Weibull Adstocking Transformation.

    This transformation is similar to geometric adstock transformation but has more
    degrees of freedom, adding more flexibility.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import WeibullType, weibull_adstock
        plt.style.use('arviz-darkgrid')

        spend = np.zeros(50)
        spend[0] = 1

        shapes = [0.5, 1., 1.5, 5.]
        scales = [10, 20, 40]
        modes = [WeibullType.PDF, WeibullType.CDF]

        fig, axes = plt.subplots(
            len(shapes), len(modes), figsize=(12, 8), sharex=True, sharey=True
        )
        fig.suptitle("Effect of Changing Weibull Adstock Parameters", fontsize=16)

        for m, mode in enumerate(modes):
            axes[0, m].set_title(f"Mode: {mode.value}")

            for i, shape in enumerate(shapes):
                for j, scale in enumerate(scales):
                    adstock = weibull_adstock(
                        spend, lam=scale, k=shape, type=mode, l_max=len(spend)
                    ).eval()

                    axes[i, m].plot(
                        np.arange(len(spend)),
                        adstock,
                        label=f"Scale={scale}",
                        linestyle="-",
                    )

        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="center right",
            bbox_to_anchor=(1.2, 0.85),
        )

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()



    Parameters
    ----------
    x : tensor
        Input tensor.
    lam : float, by default 1.
        Scale parameter of the Weibull distribution. Must be positive.
    k : float, by default 1.
        Shape parameter of the Weibull distribution. Must be positive.
    l_max : int, by default 12
        Maximum duration of carryover effect.
    axis : int
        The axis of ``x`` along witch to apply the convolution
    mode : ConvMode, optional
        The convolution mode determines how the convolution is applied at the boundaries
        of the input signal, denoted as "x." The default mode is ConvMode.Before.

        - ConvMode.After: Applies the convolution with the "Adstock" effect, resulting in a trailing decay effect.
        - ConvMode.Before: Applies the convolution with the "Excitement" effect, creating a leading effect
            similar to the wow factor.
        - ConvMode.Overlap: Applies the convolution with both "Pull-Forward" and "Pull-Backward" effects,
            where the effect overlaps with both preceding and succeeding elements.
    type : WeibullType or str, by default WeibullType.PDF
        Type of Weibull adstock transformation to be applied (PDF or CDF).

    Returns
    -------
    tensor
        Transformed tensor based on Weibull adstock transformation.
    """
    lam = pt.as_tensor(lam)[..., None]
    k = pt.as_tensor(k)[..., None]
    t = pt.arange(l_max, dtype=x.dtype) + 1

    if type == WeibullType.PDF:
        w = pt.exp(pm.Weibull.logp(t, k, lam))
        w = (w - pt.min(w, axis=-1)[..., None]) / (
            pt.max(w, axis=-1)[..., None] - pt.min(w, axis=-1)[..., None]
        )
    elif type == WeibullType.CDF:
        w = 1 - pt.exp(pm.Weibull.logcdf(t, k, lam))
        shape = (*w.shape[:-1], w.shape[-1] + 1)
        padded_w = pt.ones(shape, dtype=w.dtype)
        padded_w = pt.set_subtensor(padded_w[..., 1:], w)
        w = pt.cumprod(padded_w, axis=-1)
    else:
        raise ValueError(f"Wrong WeibullType: {type}, expected of WeibullType")
    return batched_convolution(x, w, axis=axis, mode=mode)
