# Megascale

## Overview

The change in thermodynamic folding stability ("ddG") under a single mutation has been
measured for numerous proteins and is provided in `data/data.csv`. This data is
part of the megascale dataset.

This repository holds the code to train a 1-dimensional CNN model on the wild type's
and mutant's amino acid sequences to predict ddG. We apply the Spearman correlation
as a metric to evaluate the quality and usefulness of a model.

The amino acids are embedded using either (1) a one-hot encoding or 
(2) the three-dimensional version of the ZScales encoding as described in 
[this](https://doi.org/10.1093/mp/sst148) work by Wieslander et al.
For one position of the sequence, we concatenate the embedding for the residue in 
the wild type sequence and the one for the residue in the mutated sequence.

The CNN is not applied to the full sequence, but only to a subsequence cut 
symmetrically around the mutated position, and if necessary, padded at its beginning
or end.

## Installation

Create your own virtual Python environment

```python
python -m venv megascale_venv
```

and activate it:

```python
source megascale_venv/bin/activate
```

As this repository uses a package from a private repository, 
install the requirements like this:

```python
pip install -r requirements.txt --extra-index-url https://__token__:<insta_fs_token>@gitlab.com/api/v4/projects/47354289/packages/pypi/simple
```

The value for `<insta_fs_token>` is stored in InstaDeep’s 1Password 
under the name “InstaFS Package Registry Token”.

## Execution

To run the main script, which is the model training, execute:

```python
python main.py
```

The configuration can be adapted in the `config/main.yaml` file.
The logs are not only written to stdout, but also to an `outputs` directory.
The final model parameters will be uploaded to S3 (location can be specified in
the config). Make sure to set the correct environment variables to authenticate to S3.


## Testing

There's an example unit test under the `tests/` directory.

If you have set up conda/poetry correctly and have installed `pytest`,
then `pytest tests/` should work.

Otherwise, you might need to modify the `PYTHONPATH` environment variable:
```bash
pip install -r requirements-test.txt
export PYTHONPATH=`pwd`
pytest tests/
```
