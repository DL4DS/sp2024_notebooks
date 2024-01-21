# Deep Learning for Data Science Spring 2024 Jupyter Notebooks

This is the repository for the Jupyter notebooks that are assigned
as part of the Spring 2024 edition of Deep Learning for Data Science

You basically have two choices for running the notebooks.

## Running Notebooks in Google Colab

This will usually be the easiest option for you.

Near the top of the notebook you should see a badge that looks like<br>
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>

You can click on that badge ihn the notebook (the one here won't do anyting) and
the notebook should open in Google Colab.

Once there, you can save to your Google Drive, and then any changes you make
will be saved, and you can come back to it later by opening from your 
Google Drive.

If you navigate to "My Drive" on the left side navigation bar, you should see
a folder called "Colab Notebooks". It should be in there somewhere.

## Running Notebooks Locally

You can also clone or fork-then-clone this repo and then run and edit the 
notebooks in your local environment.

It's always best to setup a virtual python environment first.

### MacOS and Linux

On MacOS and Linux:

```shell
python3 -m venv .venv # create a new python3 virtual environment called '.venv'

source .venv/bin/activate  # activate the environment

pip list # list packages in your virtual environment

# It should only show two packages, and it may suggest that you update pip.
pip install --upgrade pip 

# We'll try to keep the requirements.txt file up to date.
# You may have to rerun this command for subsequent notebooks since
# additional packages may be required.
pip install -r requirements.txt
```

### Windows

Let the instructors know if you want to run on windows and want instructions on how to do that.

### BU SCC

Let the instructors know if you want to run on BU SCC and want instructions
on how to do that.
