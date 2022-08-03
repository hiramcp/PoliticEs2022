# PoliticES task at IberLEF@SEPLN 2022
## Paper: Low-dimensional Stacking Model for Political Ideology Profiling
This repository contains our code and models for solving the _PoliticEs: Spanish Author Profiling for Political Ideology_ carried out at _IberLEF@SEPLN 2022_.

>INFOTEC-LaBD at PoliticES 2022: Low-dimensional Stacking Model for Political Ideology Profiling. Hiram Cabrera, Eric S. Tellez and Sabino Miranda. IberLEF@SEPL 2022 (URL).

Our methodology stacks several low-dimensional representations that can be used to visualize the dataset and as the input dataset for a classifier. This repository shows how to apply our approach to user profiling. Using a bunch of Twitter users' messages, we created these models and predicted gender, profession, and binary and multiclass ideology tasks.


## About the implementation
There are two different source code files, one for each kind of classifier:

- `src/task_solver.py` contains the Gradient Boosting implementation.
- `src/task_solver_svm.py` includes both SVM Linear and SVM RBF approaches.

The text is preprocessed and vectorized using the [`TextSearch.jl`](https://github.com/sadit/TextSearch.jl) and the low-dimensional projection is performed with [`SimSearchManifoldLearning.jl`](https://github.com/sadit/SimSearchManifoldLearning.jl). The notebook `src/Iberlef2022-F.ipynb` exemplifies the preprocessing-projection pipeline.

## About the data
This repository provides only embedding vectors and labels; the original data can be retrieved from the PoliticES task at [codalab](https://codalab.lisn.upsaclay.fr/competitions/1948).

## Disclaimer
This implementation is a work-in-progress. If you encounter any issues, please create an issue or make a pull request.

