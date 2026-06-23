# Muon Flux Unfolding

## Setup

The `pyproject.toml` file contains all dependencies. 
The project can be setup, e.g. with the `uv` python project manager like this
```bash
cd muflux
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
```

## Project Structure

- `utils/fit.py` contains the fitting class to run the maximum likelihood fit
- `utils/helpers.py` contains utility functions to get and plot the migration matrix, efficiency, ...
- `unfolding_example.ipynb` shows an example unfolding for a toy dataset
- `flux_data.ipynb` is an auxiliary script, to save the hard-coded flux values as numpy arrays
- `muon_flux.ipynb` shows the unfolding of the above muon flux
