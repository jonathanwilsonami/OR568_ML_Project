load_project_packages <- function() {
  required_packages <- c(
    "tidyverse",
    "janitor",
    "lubridate",
    "skimr",
    "caret",
    "glmnet",
    "broom",
    "ggplot2",
    "kableExtra",
    "dplyr",
    "tidyr",
    "stringr",
    "igraph",
    "ggraph",
    "tidygraph",
    "tibble",
    "knitr",
    "arrow"
  )

  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]

  if (length(missing_packages) > 0) {
    stop(
      paste0(
        "Missing required R packages: ",
        paste(missing_packages, collapse = ", "),
        "\nInstall them first before rendering."
      )
    )
  }

  invisible(lapply(required_packages, library, character.only = TRUE))
}

load_flight_data <- function(
  url = "https://or568-flight-delay-data-411750981882-us-east-1-an.s3.us-east-1.amazonaws.com/enriched_flights_2019.parquet",
  file_name = NULL,
  lazy = TRUE
) {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' required. Install with install.packages('arrow')")
  }

  resolve_url <- function(url, file_name = NULL) {
    if (is.null(file_name) || !nzchar(file_name)) {
      return(url)
    }

    base_url <- sub("[^/]+$", "", url)
    paste0(base_url, file_name)
  }

  resolved_url <- resolve_url(url, file_name)

  project_dir <- Sys.getenv("QUARTO_PROJECT_DIR", unset = getwd())
  data_dir <- file.path(project_dir, "shared-notebooks", "data")

  if (!dir.exists(data_dir)) {
    dir.create(data_dir, recursive = TRUE)
  }

  local_file_name <- basename(resolved_url)
  if (!nzchar(local_file_name)) {
    stop("Could not infer a local file name from the URL.")
  }

  local_file <- file.path(data_dir, local_file_name)

  if (!file.exists(local_file)) {
    message(sprintf("Downloading dataset from: %s", resolved_url))
    download.file(resolved_url, destfile = local_file, mode = "wb")
  }

  if (lazy) {
    return(arrow::open_dataset(local_file, format = "parquet"))
  } else {
    return(arrow::read_parquet(local_file))
  }
}