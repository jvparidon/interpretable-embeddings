import subprocess as sp
import numpy as np
import pandas as pd
import seaborn as sns
from subs2vec.neighbors import compute_nn
from subs2vec.utensils import log_timer
from copy import deepcopy
from sklearn.decomposition import PCA, FastICA
from statsmodels.multivariate.factor import Factor
from statsmodels.multivariate.factor_rotation import rotate_factors
from IPython.display import display, display_markdown


def display_md(md, **kwargs):
    return display_markdown(md, raw=True, **kwargs)


def convert_notebook(title, output='html'):
    convert = sp.run(f'jupyter nbconvert {title}.ipynb --to {output} --output {title}.{output}'.split(' '))
    if convert.returncode == 0:
        display_md(f'Jupyter notebook `{title}` converted successfully.')
    else:
        display_md(f'Error: encountered problem converting Jupyter notebook `{title}`')
    return convert


def download(fname):
    dl = sp.run(f'wget {fname}'.split(' '))
    if dl.returncode == 0:
        display_md(f'Download of `{fname}` succesful.')
    else:
        display_md(f'Download of `{fname}` failed.')
    return dl


def correlations(vecs, d):
    corrs = vecs.as_df().corr()
    heatmap = sns.heatmap(corrs, vmin=-1, vmax=1)
    corrs = corrs - np.eye(d)
    summary = pd.DataFrame(corrs.values.ravel(), columns=['summary']).describe()
    return corrs, summary, heatmap


@log_timer
def pca_neighbors(vecs, d, num_neighbors=5):
    # find principal components
    pca = PCA(d)
    components = pca.fit(vecs.vectors).components_
    
    # find neighbors
    labels = np.array(list(range(d)))
    neighbors = compute_nn(vecs, components, labels, num_neighbors, whole_matrix=True)
    
    return neighbors, components


@log_timer
def ica_neighbors(vecs, d, num_neighbors=5):
    # find principal components
    ica = FastICA(d)
    components = ica.fit(vecs.vectors).components_
    
    # find neighbors
    labels = np.array(list(range(d)))
    neighbors = compute_nn(vecs, components, labels, num_neighbors, whole_matrix=True)
    
    return neighbors, components


@log_timer
def fa_neighbors(vecs, d, num_neighbors=5, rotation=None, method='pa', rotate_args=[]):
    # find latent factors
    fa = Factor(vecs.vectors, d, method=method)
    loadings = fa.fit().loadings
    padding = np.zeros((vecs.vectors.shape[1], d - loadings.shape[1]))
    loadings = np.hstack([loadings, padding])
    
    # rotate factors
    if rotation is not None:
        loadings, transformation = rotate_factors(loadings, rotation, *rotate_args)
    
    # find neighbors
    labels = np.array(list(range(d)))
    neighbors = compute_nn(vecs, loadings.T, labels, num_neighbors, whole_matrix=True)
    
    return neighbors, loadings


@log_timer
def filter_vecs(vecs, filter_words):
    filtered_vecs = deepcopy(vecs)
    filtered_vecs.vectors = filtered_vecs.vectors[np.isin(filtered_vecs.words, filter_words)]
    filtered_vecs.words = filtered_vecs.words[np.isin(filtered_vecs.words, filter_words)]
    filtered_vecs.n = len(filtered_vecs.words)
    display_md(f'Filtered {vecs.n} vectors, {filtered_vecs.n} remaining.')
    return filtered_vecs
