#cleaning Crew 
#commented out - already run
#install.packages("janitor") # Great for cleaning names
#library(tidyverse)
#library(janitor)


# This turns "Incident Date" into "incident_date"
aviation_data <- aviation_data %>% 
  clean_names()


#view data
view(aviation_data)

#view head and tail of data
head(aviation_data)
tail(aviation_data)

#to see 1st 10 rows
head(aviation_data, 10)

#summary of data
summary(aviation_data)

#clean up crew was installed
#rename columns in database
library(dplyr)
#rename one or more specific columns
aviation_data <- aviation_data %>%
  rename(
    ACN = x1)

aviation_data <- aviation_data %>%
  rename(
    date = time_2,
    local_time_of_day = time_3,
    airport_code = place_4, 
    state = place_5, 
    relative_position_angle = place_6,
    relative_position_distance = place_7, 
    altitude_AGL = place_8, 
    altitude_MSL = place_9, 
    latitude_longitude = place_10, 
    flight_conditions = environment_11, 
    weather_elements = environment_12,
    work_environment_factor = environment_13,
    light = environment_14,
    ceiling = environment_15,
    RVR = environment_16,
    ATC = aircraft_1_17,
    aircraft_operator = aircraft_1_18,
    make_model_name = aircraft_1_19,
    aircraft_zone = aircraft_1_20,
    crew_size = aircraft_1_21)

#rename tables new name = old name
aviation_data <- aviation_data %>%
  rename(
    operating_under_FAR_Part = aircraft_1_22)
    

#check where the error was    
rlang::last_trace()
rlang::last_trace(drop = FALSE)

# Replaces all dots with underscores in all column names
aviation_data <- aviation_data %>%
  rename_with(~ gsub("\\.", "_", .x))

#checking for duplicates
# This counts how many rows are exact copies of another row
aviation_data %>%
  duplicated() %>%
  sum()

# This creates a table showing which IDs appear more than once
aviation_data %>%
  get_dupes(report_id)


# Shows only the rows that are duplicates
aviation_data %>%
  filter(duplicated(.))

# Keeps only the unique rows
aviation_data <- aviation_data %>%
  distinct()

# Summary of missing values per column
colSums(is.na(aviation_data))

# Option A: Remove every row that has ANY missing value (Careful!)
# clean_data <- aviation_data %>% drop_na()

# Option B: Replace NAs in a specific column with "Unknown" or 0
aviation_data <- aviation_data %>%
  mutate(place_6 = replace_na(place_6, "Unknown"))

#remove row 1
library(dplyr)

# Remove the first row
aviation_data <- aviation_data %>%
  slice(-1)

#remove columns with no data
library(janitor)

# This removes any column where every single cell is NA
aviation_data <- aviation_data %>%
  remove_empty("cols")

# If you also want to remove rows that are completely empty:
# aviation_data <- aviation_data %>% remove_empty(c("rows", "cols"))


#group columns component_55 and component_52 - show failures

library(dplyr)

# 1. Create a summary table
Failure_summary <- aviation_data %>%
  # Filter for only the rows that contain your keywords in the issue column
  filter(grepl("Improperly Operated|malfunctioning|Failed", component_55, ignore.case = TRUE)) %>%
  
  # Group by both the component and the specific issue
  group_by(component_52, component_55) %>%
  
  # Count the occurrences
  summarise(count = n(), .groups = 'drop') %>%
  
  # Sort by the most frequent malfunctions
  arrange(desc(count))

# 2. View the result
print(malfunction_summary)

#to see specific namber of rows
# Replace 'malfunction_summary' with your table name
malfunction_summary %>% print(n = 50)

#to see every row
malfunction_summary %>% print(n = Inf)


library(dplyr)

# Create a new table with only malfunctioning issues
malfunctioning_only <- aviation_data %>%
  filter(component_55 == "Malfunctioning") %>%
  group_by(component_52) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# View the result
View(malfunctioning_only)


library(dplyr)

# Create a new table with only failed issues
failed_only <- aviation_data %>%
  filter(component_55 == "Failed") %>%
  group_by(component_52) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# View the result
View(failed_only)


library(dplyr)

# Create a new table with only Improperly Operated issues
improper_operation_only <- aviation_data %>%
  filter(component_55 == "Improperly Operated") %>%
  group_by(component_52) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# View the result
View(improper_operation_only)



