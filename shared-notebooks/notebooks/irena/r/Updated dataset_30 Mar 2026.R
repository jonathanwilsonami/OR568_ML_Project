# ============================================================
# Step 1. Load Packages
# ============================================================
pacman::p_load(
  tidyverse, janitor, lubridate, skimr,
  caret, glmnet, broom,
  ggplot2, knitr, kableExtra,
  arrow, corrplot
)

# ============================================================
# Step 2. Load Parquet Data
# ============================================================
flights_raw <- open_dataset("C:/Users/ZyrIr/OneDrive/Desktop/flights_canonical_2019.parquet") %>%
  collect() %>%
  clean_names()

# ============================================================
# Step 3. The "Elite" Cleaning Pipeline
# ============================================================
df_final <- flights_raw %>%
  # Filter for your NYC hubs
  filter(origin == "BWI", dest %in% c("JFK", "EWR")) %>%
  
  # SELECT: We must include the "Elapsed Time" columns for the Honesty Check!
  select(
    arr_delay, dep_delay, carrier_delay, weather_delay, 
    dep_hour_local, dep_weekday_local, day_of_week, month, 
    dep_temp_c, dep_wind_speed_m_s, dep_ceiling_height_m,
    reporting_airline, taxi_out,
    prev_arr_delay, turnaround_minutes, aircraft_leg_number_day,
    crs_elapsed_time, actual_elapsed_time # <--- ADDED THESE TWO
  ) %>%
  
  # CLEAN: Handle NAs and create Factors
  mutate(
    prev_arr_delay = replace_na(prev_arr_delay, 0),
    weather_delay  = replace_na(weather_delay, 0),
    carrier_delay  = replace_na(carrier_delay, 0)
  ) %>%
  filter(!is.na(arr_delay), !is.na(dep_delay)) %>%
  
  # FEATURE ENGINEERING
  mutate(
    is_delayed   = as.factor(ifelse(arr_delay >= 15, "Yes", "No")),
    is_weekend   = as.factor(ifelse(day_of_week %in% c(6, 7), "Yes", "No")),
    is_rush_hour = as.factor(ifelse(dep_hour_local %in% c(7, 8, 16, 17, 18), "Yes", "No")),
    reporting_airline = as.factor(reporting_airline),
    # Calculate the Scheduling Buffer (Honesty Check)
    buffer = crs_elapsed_time - actual_elapsed_time
  )

# Health Check
skim(df_final)

# ============================================================
# Step 4. Visualizations
# ============================================================

# 1. Correlation Matrix
df_num <- df_final %>% select(where(is.numeric)) %>% na.omit()
corrplot(cor(df_num), method = "color", addCoef.col = "black", 
         title = "Predictor Correlations", mar=c(0,0,1,0), tl.cex = 0.7)

# 2. Linear Power (R-Squared Visual)
ggplot(df_final, aes(x = dep_delay, y = arr_delay)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "BWI to JFK/EWR: Linear Prediction Power",
       x = "Departure Delay (min)", y = "Arrival Delay (min)") +
  theme_minimal()

# 3. Wind Impact
ggplot(df_final, aes(x = cut(dep_wind_speed_m_s, 5), y = arr_delay)) +
  geom_boxplot(fill = "lightblue", outlier.shape = NA) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(title = "Impact of Wind Speed on Arrival Delay",
       x = "Wind Speed (m/s) Bins", y = "Arrival Delay (min)") +
  theme_minimal()

# 4. The "Tired Plane" Leg Effect
df_legs <- df_final %>%
  filter(!is.na(aircraft_leg_number_day), aircraft_leg_number_day <= 6)

ggplot(df_legs, aes(x = as.factor(aircraft_leg_number_day), y = arr_delay)) +
  geom_boxplot(fill = "firebrick", alpha = 0.6, outlier.alpha = 0.1) +
  coord_cartesian(ylim = c(-15, 120)) +
  labs(title = "The Propagated Delay Effect: Arrival Delay by Flight Leg",
       x = "Flight Leg Number of the Day", y = "Arrival Delay (Minutes)") +
  theme_minimal()

# 5. Scheduling Honesty Summary
honesty_summary <- df_final %>%
  group_by(reporting_airline) %>%
  summarise(
    avg_buffer = mean(buffer, na.rm = TRUE),
    on_time_rate = mean(arr_delay <= 0, na.rm = TRUE) * 100
  )
print(honesty_summary)


#Finding big hubs
df_clean <- flights_raw %>% 
  filter(origin == "BWI", dest %in% c("JFK", "EWR"))
# Count how many flights go to each destination
hub_counts <- flights_raw %>%
  group_by(dest) %>%
  summarise(
    total_flights = n(),
    unique_airlines = n_distinct(reporting_airline),
    avg_taxi_in = mean(taxi_in, na.rm = TRUE)
  ) %>%
  arrange(desc(total_flights))

# View the Top 10
head(hub_counts, 10)

# Plotting the Top 10 Hubs from your hub_counts summary
ggplot(head(hub_counts, 10), aes(x = reorder(dest, total_flights), y = total_flights, fill = total_flights)) +
  geom_col() +
  coord_flip() +  # Horizontal bars are easier to read
  scale_fill_gradient(low = "cadetblue2", high = "cadetblue4") +
  labs(
    title = "Top 10 Destination Hubs in the 2019 Dataset",
    subtitle = "Analysis of total flight arrivals by airport code",
    x = "Airport Code",
    y = "Total Number of Flights"
  ) +
  theme_minimal()



#Hubs from BWI
# Step 1: Filter for BWI and count the top 10 destinations
hub_data <- flights_raw %>%
  filter(origin == "BWI") %>%
  count(dest, sort = TRUE) %>%
  top_n(10, n)

# Step 2: Create the horizontal bar graph
ggplot(hub_data, aes(x = reorder(dest, n), y = n, fill = n)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +  # Makes it easier to read the airport codes
  scale_fill_gradient(low = "skyblue", high = "dodgerblue4") +
  labs(
    title = "Top 10 Destination Hubs from BWI (2019)",
    subtitle = "Analysis based on total flight frequency",
    x = "Airport Code (Destination)",
    y = "Total Number of Flights"
  ) +
  theme_minimal()

# Calculate which airlines 'pad' their schedules the most
honesty_check <- df_final %>%
  mutate(buffer = crs_elapsed_time - actual_elapsed_time) %>%
  group_by(reporting_airline) %>%
  summarise(
    avg_padding_mins = mean(buffer, na.rm = TRUE),
    avg_arrival_delay = mean(arr_delay, na.rm = TRUE),
    flight_count = n()
  ) %>%
  arrange(desc(avg_padding_mins))

# View the results
kable(honesty_check, caption = "Airline Scheduling Honesty (Padding vs. Delay)") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))


# Count total flights per airline across the entire dataset
airline_counts <- flights_raw %>%
  group_by(reporting_airline) %>%
  summarise(total_flights = n()) %>%
  arrange(desc(total_flights))

# Display the top results
head(airline_counts)

# Plot the top airlines
ggplot(head(airline_counts, 10), aes(x = reorder(reporting_airline, total_flights), y = total_flights, fill = total_flights)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = "plum1", high = "purple4") +
  labs(
    title = "Airline Market Share (2019 Dataset)",
    subtitle = "Total flight volume by reporting carrier",
    x = "Airline Code",
    y = "Total Number of Flights"
  ) +
  theme_minimal()



# Define your target airlines
target_airlines <- c("WN", "DL", "AA", "OO", "UA", "YX", "MQ", "B6", "OH", "AS")

# Calculate Scheduling Buffer (Padding)
airline_buffer_analysis <- df_final %>%
  filter(reporting_airline %in% target_airlines) %>%
  # Buffer = Scheduled Duration - Actual Duration
  mutate(buffer_mins = crs_elapsed_time - actual_elapsed_time) %>%
  group_by(reporting_airline) %>%
  summarise(
    avg_padding = mean(buffer_mins, na.rm = TRUE),
    median_padding = median(buffer_mins, na.rm = TRUE),
    on_time_rate = mean(arr_delay <= 0, na.rm = TRUE) * 100,
    avg_arr_delay = mean(arr_delay, na.rm = TRUE),
    flight_count = n()
  ) %>%
  arrange(desc(avg_padding))

# Display the table formatted for your report
airline_buffer_analysis %>%
  knitr::kable(caption = "Airline Scheduling Buffer Analysis (2019)",
               digits = 2,
               col.names = c("Airline", "Avg Padding (min)", "Median Padding", "On-Time %", "Avg Delay", "Flights"))


# Calculate Buffer for ALL airlines in the dataset
all_airline_buffers <- flights_raw %>%
  # Ensure we have the timing columns
  filter(!is.na(crs_elapsed_time), !is.na(actual_elapsed_time)) %>%
  # Buffer = Scheduled Time - Actual Time
  mutate(buffer_mins = crs_elapsed_time - actual_elapsed_time) %>%
  group_by(reporting_airline) %>%
  summarise(
    avg_buffer_mins = mean(buffer_mins, na.rm = TRUE),
    median_buffer   = median(buffer_mins, na.rm = TRUE),
    flight_count    = n(),
    on_time_arrival_rate = mean(arr_delay <= 0, na.rm = TRUE) * 100
  ) %>%
  # Only include airlines with a significant number of flights (e.g., > 100)
  filter(flight_count > 100) %>%
  arrange(desc(avg_buffer_mins))

# Display the Top 15 "Padders" vs "Lean" Operators
print(all_airline_buffers)

ggplot(all_airline_buffers, aes(x = reorder(reporting_airline, avg_buffer_mins), y = avg_buffer_mins)) +
  geom_segment(aes(xend = reporting_airline, yend = 0), color = "grey") +
  geom_point(size = 4, aes(color = avg_buffer_mins)) +
  scale_color_gradient(low = "lightpink", high = "purple") +
  coord_flip() +
  labs(
    title = "Which Airlines Build the Most 'Rest Time' into their Schedules?",
    subtitle = "Average Buffer (Scheduled Time minus Actual Flight Time)",
    x = "Reporting Airline",
    y = "Average Buffer (Minutes)"
  ) +
  theme_minimal()


# Compare Weekday vs Weekend Buffers
weekend_comparison <- df_final %>%
  mutate(
    # Buffer = Scheduled - Actual
    buffer_mins = crs_elapsed_time - actual_elapsed_time,
    day_type = ifelse(day_of_week %in% c(6, 7), "Weekend", "Weekday")
  ) %>%
  group_by(reporting_airline, day_type) %>%
  summarise(
    avg_buffer = mean(buffer_mins, na.rm = TRUE),
    avg_arr_delay = mean(arr_delay, na.rm = TRUE),
    flight_count = n()
  ) %>%
  arrange(reporting_airline, day_type)

# View the results
print(weekend_comparison)

ggplot(weekend_comparison, aes(x = reporting_airline, y = avg_buffer, fill = day_type)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("Weekday" = "steelblue", "Weekend" = "orange")) +
  labs(
    title = "Airline Scheduling Strategy: Weekdays vs. Weekends",
    subtitle = "Average Buffer (Minutes) by Reporting Carrier",
    x = "Airline Code",
    y = "Avg Buffer (Minutes)"
  ) +
  theme_minimal()



# Define 2019 Holiday Windows
df_holiday <- df_final %>%
  mutate(
    month_day = format(as.Date(flight_date), "%m-%d"),
    is_holiday = case_when(
      # Thanksgiving Period 2019
      month_day >= "11-22" & month_day <= "12-02" ~ "Holiday",
      # Winter Holiday Period 2019
      month_day >= "12-19" | month_day <= "01-05" ~ "Holiday",
      TRUE ~ "Non-Holiday"
    )
  )

# Compare Performance
holiday_stats <- df_holiday %>%
  group_by(is_holiday, reporting_airline) %>%
  summarise(
    avg_delay = mean(arr_delay, na.rm = TRUE),
    on_time_rate = mean(arr_delay <= 0, na.rm = TRUE) * 100,
    avg_taxi_out = mean(taxi_out, na.rm = TRUE)
  ) %>%
  arrange(reporting_airline, is_holiday)

print(holiday_stats)


#########################################################
# Load the library
pacman::p_load(rpart, rpart.plot)

# 1. Train the Decision Tree
# We use 'is_delayed' as the target (Yes/No)
tree_model <- rpart(is_delayed ~ dep_delay + dep_hour_local + dep_wind_speed_m_s + 
                      aircraft_leg_number_day + reporting_airline, 
                    data = df_final, method = "class")

# 2. Visualize the Tree (The 'Logic Map')
rpart.plot(tree_model, type = 4, extra = 101, under = TRUE, cex = 0.8,
           main = "BWI-NYC Delay Decision Logic")

# Load the library
pacman::p_load(randomForest)

# 1. Train the Random Forest
# Note: Random Forest is memory intensive, so we use a subset if needed
rf_model <- randomForest(is_delayed ~ dep_delay + dep_hour_local + dep_wind_speed_m_s + 
                           prev_arr_delay + taxi_out + aircraft_leg_number_day, 
                         data = df_final, ntree = 100, importance = TRUE)

# 2. Check the Variable Importance
# This shows you which feature was the 'MVP' of the model
importance_df <- as.data.frame(importance(rf_model))
varImpPlot(rf_model, main = "Which Features Predict Delays Best?")

# Predict using the Random Forest
rf_preds <- predict(rf_model, df_final)

# Create the Matrix
confusionMatrix(rf_preds, df_final$is_delayed)


#######################
# Step 1: Create a specialized 'Tree Data' set with NO NAs
# We must remove NAs from the predictors we plan to use
df_tree <- df_final %>%
  filter(!is.na(dep_wind_speed_m_s), 
         !is.na(aircraft_leg_number_day),
         !is.na(dep_hour_local)) %>%
  # Random Forest requires the target to be a Factor
  mutate(is_delayed = as.factor(is_delayed))

# Step 2: Re-run the Random Forest on the CLEANED data
# We'll use 100 trees to keep it fast for your BWI-NYC subset
rf_model <- randomForest(is_delayed ~ dep_delay + dep_hour_local + 
                           dep_wind_speed_m_s + aircraft_leg_number_day, 
                         data = df_tree, 
                         ntree = 100, 
                         importance = TRUE,
                         na.action = na.roughfix) # <--- This handles tiny stray NAs automatically

# Step 3: Now run the prediction on the SAME cleaned data
rf_preds <- predict(rf_model, df_tree)

# Step 4: Final Confusion Matrix
confusionMatrix(rf_preds, df_tree$is_delayed)




###################
# Step 1: Install and load the high-speed engine
pacman::p_load(ranger)

# Step 1: Prepare the FULL dataset (Select raw columns FIRST)
df_full_tree <- flights_raw %>%
  # Select only the columns that exist in the Parquet file
  select(
    arr_delay, dep_delay, dep_hour_local, 
    dep_wind_speed_m_s, aircraft_leg_number_day, 
    distance, carrier_delay
  ) %>%
  # NOW create the 'is_delayed' column
  mutate(is_delayed = as.factor(ifelse(arr_delay >= 15, "Yes", "No"))) %>%
  # Clean up any NAs so the model doesn't crash
  drop_na()

# Step 2: Run the High-Speed Ranger Forest
pacman::p_load(ranger)

full_rf_model <- ranger(
  formula   = is_delayed ~ ., 
  data      = df_full_tree, 
  num.trees = 100,           
  importance = "impurity", 
  num.threads = parallel::detectCores() - 1, 
  verbose   = TRUE
)

# Step 3: Check the Results
print(full_rf_model)
vip_data <- sort(full_rf_model$variable.importance, decreasing = TRUE)
print(vip_data)