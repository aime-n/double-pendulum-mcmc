## Double Pendulum Project

### Requirements

- Python 3.10.14 (or any 3.10.x version) 
- `matplotlib`
- `numpy>=1.21`
- `scipy>=1.8.0`
- `torch>=1.10.2`
- `pandas>=1.4.1`
- `pymc3`
- `double-pendulum` from [this GitLab repository](https://gitlab.com/eijikawasaki/double-pendulum.git)
- `theano`

### Installation
1. **Clone the Repository:**

```bash
git clone https://github.com/aime-n/double-pendulum-mcmc.git
cd double-pendulum-project
```

2. **Set Up a Virtual Environment and Install the requirements:**

```bash
python3.10 -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
pip install -r requirements.txt
```

### Running the Project

To run the entire modeling pipeline, execute the `main.py` script. This script will sequentially run the necessary scripts in the correct order.

```bash
python main.py
```




<!-- Average Negative Log-Likelihood: 1.0955941677093506  -->
