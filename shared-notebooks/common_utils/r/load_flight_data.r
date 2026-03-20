load_flight_data <- function(
  # This function loads the flight delay dataset, downloading it from S3 if not already cached locally.
  url = "https://or568-flight-delay-data-411750981882-us-east-1-an.s3.us-east-1.amazonaws.com/enriched_flights_2019.parquet"
) {

  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' required. Install with install.packages('arrow')")
  }

  if (!requireNamespace("here", quietly = TRUE)) {
    stop("Package 'here' required. Install with install.packages('here')")
  }

  # Construct project-relative path
  data_dir <- here::here("shared-notebooks", "data")
  local_file <- file.path(data_dir, "enriched_flights_2019.parquet")

  # Create directory if needed
  if (!dir.exists(data_dir)) {
    dir.create(data_dir, recursive = TRUE)
  }

  # Download if not cached
  if (!file.exists(local_file)) {
    message("Downloading dataset from S3...")
    download.file(url, destfile = local_file, mode = "wb")
  }

  arrow::read_parquet(local_file)
}