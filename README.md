[![PyPI](https://img.shields.io/pypi/v/tsproto)](https://pypi.org/project/tsproto/)  ![License](https://img.shields.io/github/license/sbobek/tsproto)
 ![PyPI - Downloads](https://img.shields.io/pypi/dm/tsproto) [![Documentation Status](https://readthedocs.org/projects/tsproto/badge/?version=latest)](https://tsproto.readthedocs.io/en/latest/?badge=latest)
    
![](https://raw.githubusercontent.com/sbobek/tsproto/main/pix/workflow.svg)
# TSProto
Post-host prototype-based explanations with rules for time-series classifiers.

Key features:
  * Extracts interpretable prototype for any black-box model and creates a decision tree, where each node is constructed from the visual prototype
  * Integrated with SHAP explainer, as a backbone for extraction of interpretable components (However, SHAP can be replaced with any other feature-importance method)

## Install
TSProto can be installed from either [PyPI](https://pypi.org/project/tsproto/) or directly from source code from this repository.

To install form PyPI:

```
pip install tsproto
````

To install from source code:

```
git clone https://github.com/sbobek/tsproto
cd tsproto
pip install .
```

## Usage
For full examples on two illustrative cases go to:
  * Example of extracting sine wave prototype and explaining class with existence ora absence of a prototype: [Jupyter Notebook](https://github.com/sbobek/tsproto/blob/main/examples/illustrative-example-frequency.ipynb)
  * Example of extracting sine wave as a prototype end explaining class by difference in frequency of a prototype [Jupyter Notebook](https://github.com/sbobek/tsproto/blob/main/examples/illustrative-example.ipynb)

The basic usage of the TSProto assuming you have your model trained is straightforward:

``` python
from tsproto.models import *
from tsproto.utils import *

#assuming that trainX, trainy and model are given

pe = PrototypeEncoder(clf, n_clusters=2, min_size=50, method='dtw',
                      descriptors=['existance'],
                      jump=1, pen=1,multiplier=2,n_jobs=-1,
                      verbose=1)

trainX, shapclass = getshap(model=model, X=trainX, y=trainy,shap_version='deep',
                        bg_size = 1000,  absshap = True)

#The input needs to be a 3D vector: number of samples, lenght of time-series, number of dimensions (features)
trainXproto = train.reshape((trainX.shape[0], trainX.shape[1],1))
shapclassXproto = shapclass.reshape((shapclass.shape[0], shapclass.shape[1],1))

ohe_train, features, target_ohe,weights = pe.fit_transform(trainXproto,shapclassXproto)

im  = InterpretableModel()
acc,prec,rec,f1,interpretable_model = im.fit_or_predict(ohe_train, features,
                        target_ohe,
                        intclf=None, # if intclf is given, the funciton behaves as predict,
                        verbose=0, max_depth=2, min_samples_leaf=0.05,
                        weights=None)
                 
```

After the Interpretable model has been created it now can be visualised.

``` python
                       
# Visualize model
from  tsproto.plots import *

ds_final = ohe_train.copy()
dot = export_decision_tree_with_embedded_histograms(decision_tree=interpretable_model, 
                                              dataset=ds_final, 
                                              target_name='target', 
                                              feature_names=features, 
                                              filename='synthetic', 
                                              proto_encoder=pe, figsize=(6,3))

from IPython.display import SVG, Image
Image('synthetic.png')

```

![Prototype visualization](https://raw.githubusercontent.com/sbobek/tsproto/main/pix/illustrative-example.png "Title")


## Cite this work
More details on how the TSProto works and evaluation benchmarks can eb found in the following paper:

```
@article{boobek2025tsproto,
 title = {TSProto: Fusing deep feature extraction with interpretable glass-box surrogate model for explainable time-series classification},
 journal = {Information Fusion},
 pages = {103357},
 year = {2025},
 issn = {1566-2535},
 doi = {https://doi.org/10.1016/j.inffus.2025.103357},
 url = {https://www.sciencedirect.com/science/article/pii/S1566253525004300},
 author = {Szymon Bobek and Grzegorz J. Nalepa},
 keywords = {Explainable artificial intelligence, Time-series, Neurosymbolic, Deep neural networks},
 abstract = {Deep neural networks (DNNs) are highly effective at extracting features from complex data types, such as images and text, but often function as black-box models, making interpretation difficult. We propose TSProto – a model-agnostic approach that goes beyond standard XAI methods focused on feature importance, clustering important segments into conceptual prototypes—high-level, human-interpretable units. This approach not only enhances transparency but also avoids issues seen with surrogate models, such as the Rashomon effect, enabling more direct insights into DNN behavior. Our method involves two phases: (1) using feature attribution tools (e.g., SHAP, LIME) to highlight regions of model importance, and (2) fusion of these regions into prototypes with contextual information to form meaningful concepts. These concepts then integrate into an interpretable decision tree, making DNNs more accessible for expert analysis. We benchmark our solution on 61 publicly available datasets, where it outperforms other state-of-the-art prototype-based methods and glassbox models by an average of 10% in the F1 metric. Additionally, we demonstrate its practical applicability in a real-life anomaly detection case. The results from the user evaluation, conducted with 17 experts recruited from leading European research teams and industrial partners, also indicate a positive reception among experts in XAI and the industry. Our implementation is available as an open-source Python package on GitHub and PyPi.}
}
```
