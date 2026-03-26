load_flight_data <- function(
  # This function loads the flight delay dataset, downloading it from S3 if not already cached locally.
  url = "https://or568-flight-delay-data-411750981882-us-east-1-an.s3.us-east-1.amazonaws.com/enriched_flights_2019.parquet",
  file_name = NULL
) {

  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' required. Install with install.packages('arrow')")
  }

  if (!requireNamespace("here", quietly = TRUE)) {
    stop("Package 'here' required. Install with install.packages('here')")
  }

  resolve_url <- function(url, file_name = NULL) {
    if (is.null(file_name) || !nzchar(file_name)) {
      return(url)
    }

    base_url <- sub("[^/]+$", "", url)
    paste0(base_url, file_name)
  }

  infer_local_filename <- function(resolved_url) {
    file_name <- basename(resolved_url)
    if (!nzchar(file_name)) {
      stop("Could not infer a file name from the resolved URL.")
    }
    file_name
  }

  resolved_url <- resolve_url(url, file_name)

  # Construct project-relative path
  data_dir <- here::here("shared-notebooks", "data")

  # Create directory if needed
  if (!dir.exists(data_dir)) {
    dir.create(data_dir, recursive = TRUE)
  }

  local_file_name <- infer_local_filename(resolved_url)
  local_file <- file.path(data_dir, local_file_name)

  # Download if not cached
  if (!file.exists(local_file)) {
    message(sprintf("Downloading dataset from S3: %s", resolved_url))
    download.file(resolved_url, destfile = local_file, mode = "wb")
  }

  arrow::read_parquet(local_file)
}