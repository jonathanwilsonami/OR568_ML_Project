

Make R available inside Jupyter (one-time kernel registration)

Run:

R -q -e 'IRkernel::installspec(user = TRUE)'


Now Jupyter can run Python and R kernels.


quarto preview
quarto render 

# OR 568 ML Project & Shared Notebooks (Python + R + Quarto)

This repository is a simple, shared workspace for a mixed team using:

- Python notebooks
- R / Quarto reports
- VS Code, JupyterLab, and RStudio

The goal is to keep things:
- easy to run,
- easy to review in Git,
- and consistent across tools.

## Clone and open project root (from Github)
The exact commands will be found in Github. See the green Code button for the clone command and remote url.

```bash
# Go to your desired project directory and run:
git clone <REPO_URL>
cd <REPO_FOLDER>  
```

## Setups 
The following setup assumes you already have your IDE of choice installed (RStudio, JupyterLab or VSCode) and that you are already in your working directory of the project. 

If you need a different version of R or Python we can update the dependencies in the environment.yml. file.  

### One-time setup (recommended)
We use a single Conda environment for both Python and R. If you do not have conda install go here: [Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Run this in your terminal. 
```bash
conda env create -f environment.yml
conda activate ml-shared-notebooks
```

Register the R kernel for Jupyter (only once per machine). Run this in the terminal:
```bash
R -q -e 'IRkernel::installspec(user = TRUE)'
```

#### Installing and running Quarto 
See [Quarto installation guide](https://quarto.org/docs/get-started/) 
```bash
# To render 
quarto render 
# To see the site in your local browser 
quarto preview
```

## Making contributions to the repo. Follow this git workflow:
The following steps assume you already cloned the repo and have made changes that you want to push. 

1. **Add and commit** your changes locally. Do NOT push yet. 
2. **Do a git pull** to make sure you have the most recent changes from the remote repo. You can always run 
git status to see how upstream changes have progressed before pulling changes. 
3. **Resolve conflicts** - As necessary resolve any merge conflicts. 
4. **Run a git push** - when you are ready to push changes do a git push. 

Lastly, make sure your changes have been applied and that the gitlab pages site works. 

## Publish To Github Pages

I added a github workflow ci-cd to automatically push to Github pages. So when you add your changes
and push it should automatically push the quarto site too. Make sure you run quarto render to render before
pushing your changes. Make sure you are out of your conda env when your run quarto render or quarto preview. You should be using your system quarto you installed. 

If you need to manually push to Github Pages use the following command:

```bash
quarto publish gh-pages
```

This will push the quarto site to Github.