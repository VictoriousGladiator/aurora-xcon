from typing import Optional, Tuple

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_2d_unstructured_repertoire(
    repertoire_fitnesses: jnp.ndarray,
    repertoire_descriptors: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:

    grid_empty = repertoire_fitnesses == -jnp.inf

    my_cmap = cm.viridis

    # set the parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "figure.figsize": [10, 10],
    }

    mpl.rcParams.update(params)

    # create the plot object
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")
    ax.set(adjustable="box", aspect="equal")

    # aesthetic
    divider = make_axes_locatable(ax)
    ax.set_aspect("equal")

    # if the grid is empty, plot an empty grid
    if jnp.all(grid_empty):
        return fig, ax

    fitnesses = repertoire_fitnesses
    if vmin is None:
        vmin = float(jnp.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~grid_empty]))

    norm = Normalize(vmin=vmin, vmax=vmax)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    descriptors = repertoire_descriptors[~grid_empty]
    ax.scatter(
        descriptors[:, 0],
        descriptors[:, 1],
        c=fitnesses[~grid_empty],
        cmap=my_cmap,
        s=10,
        zorder=0,
    )

    return fig, ax


def plot_repertoire_embeddings(
    repertoire_fitnesses: jnp.ndarray,
    repertoire_descriptors: jnp.ndarray,
    embeddings_2d: jnp.ndarray,
    minval: float,
    maxval: float,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:
    grid_empty = (repertoire_fitnesses == -jnp.inf)
    
    if jnp.all(grid_empty):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
        axs[0].set_title("AURORA Rep. (t-SNE) [Empty]", fontsize=12, fontweight='bold', pad=10)
        axs[1].set_title("Corresponding Passive Descriptors [Empty]", fontsize=12, fontweight='bold', pad=10)
        return fig, axs

    active_fitnesses = repertoire_fitnesses[~grid_empty]
    active_embeddings = embeddings_2d[~grid_empty]
    active_descriptors = repertoire_descriptors[~grid_empty]

    if vmin is None:
        vmin = float(jnp.min(active_fitnesses))
    if vmax is None:
        vmax = float(jnp.max(active_fitnesses))
    if vmin == vmax:
        vmax += 1.0

    my_cmap = "plasma" 

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.5), facecolor='white')
    
    
    for i, a in enumerate(axs):
        a.set_facecolor("#fcfcfc")
        a.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e0e0e0', alpha=0.7, zorder=0)
        a.set_axisbelow(True)
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            a.spines[spine].set_color("#cccccc")
            a.spines[spine].set_linewidth(0.8)

    axs[0].set_title("AURORA Encoder Latent Space (t-SNE)", fontsize=12, fontweight='bold', pad=12, color="#333333")
    axs[0].set_xlabel("t-SNE Dimension 1", fontsize=10, labelpad=8, color="#555555")
    axs[0].set_ylabel("t-SNE Dimension 2", fontsize=10, labelpad=8, color="#555555")
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    axs[0].scatter(
        active_embeddings[:, 0],
        active_embeddings[:, 1],
        c=active_fitnesses,
        cmap=my_cmap,
        norm=norm,
        alpha=0.85,
        s=18,
        edgecolors='none',
        zorder=3
    )

    axs[1].set_xlim(minval, maxval)
    axs[1].set_ylim(minval, maxval)
    axs[1].set_title("Corresponding Passive Descriptors", fontsize=12, fontweight='bold', pad=12, color="#333333")
    axs[1].set_xlabel("Descriptor Dimension 1", fontsize=10, labelpad=8, color="#555555")
    axs[1].set_ylabel("Descriptor Dimension 2", fontsize=10, labelpad=8, color="#555555")

    axs[1].scatter(
        active_descriptors[:, 0],
        active_descriptors[:, 1],
        c=active_fitnesses,
        cmap=my_cmap,
        norm=norm,
        alpha=0.85,
        s=18,
        edgecolors='none',
        zorder=3
    )

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), 
        ax=axs, 
        orientation="vertical", 
        fraction=0.03, 
        pad=0.04
    )
    cbar.set_label("Fitness Value", fontsize=11, labelpad=10, fontweight='semibold', color="#333333")
    cbar.ax.tick_params(labelsize=9, color="#555555")
    cbar.outline.set_visible(False)
    
    fig.subplots_adjust(wspace=0.25)
    return fig, axs
