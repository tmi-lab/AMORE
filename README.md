
# AMORE

Automatic Model-agnostic Regional Rule Extraction (AMORE) is the implementation of our paper ["Enabling Regional Explainability by Automatic and
Model-agnostic Rule Extraction"](https://arxiv.org/abs/xxx). 

## Getting Started

These instructions will help you install a virtual environment to reproduce our experimental results in the paper.

### Prerequisites

- Python 3.9.15
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html )/ [pyenv](https://github.com/pyenv/pyenv) / [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Choose according to the environment setup you prefer)

### Installing


1. **Clone the repository**

    ```bash
    git clone https://github.com/yc14600/AMORE.git
    cd AMORE
    ```

2. **Using `requirements.txt` (for virtualenv or pyenv) to create virtual environment**



    1). **Create a virtual environment**

    Create a virtual environment by virtualenv or pyenv with specified python version 3.9.15, for example:

    ```bash
    pyenv virtualenv 3.9.15 amore_venv
    ```

    2). **Activate the virtual environment**

    ```bash
    pyenv activate amore_venv
    ```

    3). **Install the dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Using `environment.yml` (Conda)**


    1). **Create a Conda environment**

    ```bash
    conda env create -f amore_venv.yml
    ```

    2). **Activate the Conda environment**

    ```bash
    conda activate amore_venv
    ```

## Running the experiments

All experiments are Ipython notebooks in the "experiments" folder. You can run the notebooks in the following way:

After activating the virtual environment, first register the virtual environment to ipython kernel:
```bash
python -m ipykernel install --user --name=amore_venv
```
Then run jupyter notebook in the terminal and open the notebooks in the browser. For example, run the following commands in the terminal:
```bash
cd experiments
jupyter notebook
```
To specify the virtual environment for running a notebook file, click the ``Kernel`` tab and then click ``change kernel`` in the list, choose the kernel ``amore_venv``.

There are two datasets from Kaggle website, which need to be downloaded first before running the corresponding notebook. The links are as follows:
- [Diabetes dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- [Brain tumor MRI dataset](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256/)

Other datasets can be downloaded automatically by running the notebook first time.



## License

This project is licensed under the CC-BY-4.0 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

- The implementation of NeuralCDE model is adapted from [NeuralCDE](https://github.com/patrick-kidger/NeuralCDE/tree/master)
- The implementation of Integrated Gradients is adapted from [Simplex](https://github.com/JonathanCrabbe/Simplex)

