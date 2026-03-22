# OR 568 ML Project & Shared Notebooks (Python + R + Quarto)

The OR 568 Machine Learning Project site serves as a centralized platform for exploring, analyzing, and modeling flight delay dynamics using real-world aviation, weather, and operational data. Its primary purpose is to document the end-to-end data science workflow—from data ingestion and feature engineering to exploratory analysis and advanced modeling—while enabling a deeper understanding of delay propagation across the air transportation network. The site not only communicates key findings and insights but also provides a reproducible, well-structured environment for collaboration, experimentation, and continuous refinement of machine learning and statistical approaches to complex, spatio-temporal problems.

This repository is a simple, shared workspace for a mixed team using:

- Python notebooks
- R / Quarto reports
- VS Code, JupyterLab, and RStudio

The goal is to keep things:
- easy to run,
- easy to review in Git,
- and consistent across tools.

There are countless ways to structure a project, and while this setup is not perfect, it closely reflects many of the professional environments I’ve worked in. Regardless of the language—whether Java, C++, or Python—most production-grade projects share common foundations: version control workflows, clear documentation, organized code repositories, and reproducible setup processes.

Some projects incorporate advanced automation tools such as Terraform, Ansible, and CI/CD pipelines, while others require more complex build environments involving compilers, package managers like npm, or distributed computing frameworks such as EMR. On the other end of the spectrum, some projects remain intentionally lightweight, like this one.

My goal is for this project to give use a practical, hands-on introduction to how professional data science systems are structured and operated. In many ways, we can think of it as a “lightweight” or simplified version of platforms like Amazon SageMaker—designed to expose us to the core concepts without the overhead of enterprise-scale infrastructure.

## Clone and open project root (from Github)
The exact commands will be found in Github. See the green Code button for the clone command and remote url.

```bash
# Go to your desired project directory and run:
git clone <REPO_URL>
cd <REPO_FOLDER>  
```

## Setups 
The following setup assumes you already have your IDE of choice installed (RStudio, JupyterLab or VSCode) and that you are already in your working directory of the project. 

If you need a different version of R or Python we can update the dependencies in the environment.yml. file. See the section on working with conda. 

## Environment Setup (Conda + Python + R)

This project uses **Conda** to manage a unified environment for both **Python** and **R** dependencies. Follow the steps below to get started.

---

### 1. Install Conda

If you do not already have Conda installed, install one of the following:

- Miniconda (): https://docs.conda.io/en/latest/miniconda.html  

---

### 2. Create the Environment

Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate or568_ml_project
```

If you are using Jupyter, register the R kernel (only once per machine):

```bash
R -q -e "IRkernel::installspec(user = TRUE)"
```

## 3. Installing New Packages

If you need to install new packages, update the environment in a way that keeps it reproducible for the team.

### Preferred Method (Recommended)

Install packages using Conda:

```bash
# Python packages
conda install -c conda-forge <package_name>

# R packages (via conda)
conda install -c conda-forge r-<package_name>
```

Examples:

```bash
conda install -c conda-forge polars
conda install -c conda-forge r-janitor
```

### Using pip (Python only)

Only use pip if the package is not available in Conda:

```bash
pip install <package_name>
```

### Installing R packages via CRAN (fallback)

Only if not available in Conda:

```r
install.packages("package_name")
```

## 4. Updating the Environment File (IMPORTANT)

After installing new packages, update the shared `environment.yml` file:

```bash
conda env export --no-builds > environment.yml
```

---

Summary notes: 
- Prefer conda install when possible for compatibility
- Use pip only when a package is not available via Conda
- Keep environment.yml updated for team reproducibility

#### Installing and running Quarto 
See [Quarto installation guide](https://quarto.org/docs/get-started/) 
```bash
# To render - Note: Need to do this anytime you want your changes to be reflected on the site.
quarto render 
# To see the site in your local browser. Make sure you do this to check for any issues. 
quarto preview
```

## Project Overview

### Site Structure and Key Pages (Quarto Overview)

This site is built using **Quarto**, which converts `.qmd` (and `.ipynb`) files into a static website. The overall structure and navigation are defined in the `_quarto.yml` file at the root of the project. This file controls the **navbar (top menu)**, **sidebar (left navigation for notebooks)**, theme, and where the rendered site is output (`docs/` folder for GitHub Pages). 

### Key Pages You May Want to Edit

- **Home Page** → `index.qmd`  
  Main landing page of the site

- **About Page** → `about.qmd`  
  Update your personal info, team bios, and project context

- **Paper Page** → `paper.qmd`  
  Used for writing and presenting your final report

- **Slides Page** → `slides.qmd`  
  Used for presentation slides (Quarto supports reveal.js slides)

- **Shared Notebooks** → `shared-notebooks/`  
  Automatically populated based on directory structure (see previous section)

- **Images** → `images/`  
  Where the site grabs images. 

- **docs** → `docs/`  
  When the site is built or rendered (quarto render) it will place all the html, css, js etc. code into this folder. This folder is basically the site. It's what the git workflow will pick up (part of the CI/CD) and what github pages will deploy on github. The GitHub Action (Workflow) is responsible for **publishing the rendered Quarto site to GitHub Pages**. It does **not build the site**—it simply takes the already-rendered files in the `docs/` folder and pushes them to the `gh-pages` branch, which GitHub uses to host the website. 

---

### How Quarto Works (High-Level)

- Each `.qmd` or `.ipynb` file = **one page on the site**
- Quarto renders everything into the `docs/` folder (this is what GitHub Pages serves)
- The `_quarto.yml` file defines:
  - Navigation (navbar + sidebar)
  - Site layout and structure
  - Rendering behavior

### Shared Notebooks (Python + R)

The **Shared Notebooks** section is where each team member contributes their analysis and modeling work. For consistency, we are standardizing on:
- **`.ipynb` (Jupyter Notebooks)** for Python work  
- **`.qmd` (Quarto files)** for R work  

Please refer to the existing examples in the repository to understand formatting, structure, and best practices.

To add your own work, simply create and maintain your notebooks inside your **own named directory** under the appropriate language folder (e.g., `shared-notebooks/notebooks/<your-name>/python/` or `.../r/`). As long as you keep your files within your designated directory, they will be automatically picked up and rendered in the Quarto site—no additional configuration is needed. This keeps contributions organized, prevents conflicts, and ensures the site builds cleanly for everyone.

Also, don't delete the index.qmd. That was added to overcome a Windows related bug. 

### Shared Data File (S3-Based Workflow)

To keep our work consistent, reproducible, and streamlined, we are using a **single shared dataset stored in S3** as the source of truth for this project. This dataset is **not checked into the Git repository** and is instead downloaded and cached locally using the provided `load_flight_data` functions (available in both R and Python). Everyone should use this function to load data rather than manually downloading or creating separate versions of the dataset. This ensures that all analyses, models, and visualizations are built on the same foundation and that the Quarto site renders correctly for everyone.

If you need to perform **feature engineering or create new datasets**, please do so within your own scripts or pipeline code—do not create standalone data files that others cannot access. Any data files that are not reproducible or not available to the team will break the site when others try to render it. The expectation is that all transformations are reproducible from the shared dataset using code. In the future, additional datasets will also be stored in S3 and accessed in the same way, continuing this pattern of centralized, reproducible data access.

I've provided some examples of how to set this up but the following is an example (might change over time): 

```r
###################################################
# Code to Load Data from S3
###################################################
library(dplyr)
library(janitor)
notebook_dir <- dirname(knitr::current_input(dir = TRUE))

util_path <- normalizePath(
  file.path(
    notebook_dir,
    "..", "..", "..",
    "common_utils", "r", "load_flight_data.r"
  ),
  winslash = "/",
  mustWork = TRUE
)

source(util_path)

# Load data and clean column names
enriched_flights_2019 <- load_flight_data() %>%
  janitor::clean_names()

###################################################
# End Code to Load Data from S3
###################################################
```

## Git Workflow for Maintaining and Contributing to the Quarto Site

To keep the Quarto site stable and organized, all work should be done through feature branches rather than directly on `main`. Start by pulling the latest version of `main`, then create a new branch for your changes using a clear name such as `add-eda-page` or `irena_work` or even just your name to make things simple does not matter too much. Make your edits, preview the site locally with Quarto to confirm everything renders correctly, and commit your changes with meaningful commit messages. Once your work is ready, push your branch to GitHub and open a pull request into `main`. Before merging, verify that the site builds successfully, links work, code and notebooks run as expected, and no unnecessary generated files or large local datasets are included. After approval, merge into `main`, pull the updated `main` branch locally, and delete old branches that are no longer needed. This workflow helps prevent conflicts, protects the production version of the site, and makes collaboration much easier.

### Recommended Step-by-Step Workflow

```bash
# BEFORE making changes
# 1. Move to main and get the latest updates
git checkout main # This is the default branch and you may already be on it
git pull origin main # Get latest updates 

# 2. Create a new branch for your work
git checkout -b your-branch-name
# If you have already made changes you can move them to this new branch. 

# 3. Make changes to the Quarto site files
#    Example: .qmd files, _quarto.yml, scripts, images, etc.

# 4. Preview locally to verify the site builds
quarto render # Builds the site. You only need to run this once before pushing or opening the PR to confirm a clean full build. Otherwise your IDE will usually build your site. 
quarto preview # view changes locally 

# 5. Stage and commit your changes
# Add anything you do not want in git to .gitignore before you run the commands below!!! 

# IMPORTANT: Before committing, sync your branch with latest changes from main
git fetch origin
git merge origin/main
# Resolve any conflicts if they appear before continuing

git status # Shows you what things are tracked or untracked. Can use this to know what you need to track or commit. 

git add path/to/file1 path/to/file2
# or you can add everything like this. Caution: Make sure you know what you are pushing if you use git add .
git add . 

git commit -m "update descriptive message here"
# You can also just run git commit and it will enter you into an editor to write the comment. 
# If you prefer using an IDE you can do the same thing using buttons.

# 6. Push your branch to GitHub
git push -u origin your-branch-name

# 7. Open a Pull Request into main on GitHub
#    Review changes, discuss if needed, and merge after approval

# 8. After merge, update local main
git checkout main
git pull origin main

# 9. Delete the old branch locally only once the change has been made
git branch -d feature/your-branch-name
```

#### Best Practices
- Always preview the Quarto site locally before committing
- Do not commit large raw datasets, cached files, or environment-specific files
- If multiple people are editing, pull from main often to reduce merge conflicts. 

This is a general workflow. You may have to do some additional things if you get stuck. 

## Publish To Github Pages

I added a github workflow ci-cd to automatically push to Github pages. So when you add your changes and push it should automatically push the quarto site too. Note: This will only work if you are working directly on main. If you are working on your own branch your work will show up once your branch has been merged into main. Make sure you run quarto render to render before pushing your changes.  

If you need to manually push to Github Pages use the following command:

```bash
quarto publish gh-pages
```

This will push the quarto site to Github.