# Packages -------------------
library(arrow)
library(dplyr)
library(janitor)
library(lubridate)
library(ggplot2)
library(tidyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)

library(xgboost)
library(scales)
library(maps)
library(ggrepel)
library(nycflights13)

# Setup ---------------------------------------------------------------------


# Import file
parquet_file <- "flights_canonical_2019.parquet"

# US holidays for 2019
US_HOLIDAYS_2019 <- as.Date(c(
  "2019-01-01",
  "2019-01-21",
  "2019-02-18",
  "2019-05-27",
  "2019-07-04",
  "2019-09-02",
  "2019-10-14",
  "2019-11-11",
  "2019-11-28",
  "2019-12-25"
))

holiday_labels <- c(
  "New Year's Day",
  "MLK Day",
  "Presidents Day",
  "Memorial Day",
  "Independence Day",
  "Labor Day",
  "Columbus Day",
  "Veterans Day",
  "Thanksgiving",
  "Christmas"
)

days_before <- 3
days_after  <- 3

# Holiday-window lookup table
holiday_window_tbl <- purrr::map2_dfr(
  US_HOLIDAYS_2019,
  holiday_labels,
  ~ tibble(
    holiday_date = .x,
    holiday_name = .y,
    dep_date_local = seq(.x - days_before, .x + days_after, by = "day")
  )
) %>%
  distinct(dep_date_local, .keep_all = TRUE)

holiday_window_dates <- holiday_window_tbl$dep_date_local

# Lazy load
flights_ds <- open_dataset(parquet_file)

flights_holiday_lazy <- flights_ds %>%
  select(
    ArrDelay,
    ArrDel15,
    Month,
    DayOfWeek,
    Reporting_Airline,
    Origin,
    Dest,
    Distance,
    dep_hour_local,
    dep_temp_c,
    dep_wind_speed_m_s,
    dep_ceiling_height_m,
    prev_arr_delay,
    prev_dep_delay,
    turnaround_minutes,
    prev_arr_late_15,
    dep_date_local,
    is_cancelled,
    is_diverted
  ) %>%
  filter(
    is_cancelled == FALSE,
    is_diverted == FALSE,
    dep_date_local %in% holiday_window_dates
  )

# Collect reduced data into memory
flights_holiday <- flights_holiday_lazy %>%
  collect() %>%
  clean_names()

# Join holiday names and create relative-day feature
holiday_window_tbl_clean <- holiday_window_tbl %>%
  clean_names()

flights_holiday <- flights_holiday %>%
  mutate(dep_date_local = as.Date(dep_date_local)) %>%
  left_join(
    holiday_window_tbl_clean,
    by = "dep_date_local"
  ) %>%
  mutate(
    days_from_holiday = as.integer(dep_date_local - holiday_date),
    delayed_15 = as.integer(arr_del15),
    month = as.factor(month),
    day_of_week = as.factor(day_of_week),
    reporting_airline = as.factor(reporting_airline),
    origin = as.factor(origin),
    dest = as.factor(dest),
    prev_arr_late_15 = as.factor(prev_arr_late_15)
  )

# Checks
glimpse(flights_holiday)
summary(flights_holiday)

# Rows kept
nrow(flights_holiday)
# Plot Theme ------------------------------------------------------

report_theme <- theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0),
    plot.subtitle = element_text(
      size = 11,
      color = "gray30",
      margin = ggplot2::margin(b = 10)
    ),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(color = "gray20"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    plot.caption = element_text(size = 9, color = "gray40"),
    plot.margin = ggplot2::margin(10, 15, 10, 10)
  )

# Summaries ----------------

# Average delay by holiday
delay_by_holiday <- flights_holiday %>%
  group_by(holiday_name) %>%
  summarise(
    n = n(),
    avg_arr_delay = mean(arr_delay, na.rm = TRUE),
    pct_delayed_15 = mean(delayed_15, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_arr_delay))

print(delay_by_holiday)

# Average delay by day relative to holiday
delay_by_relative_day <- flights_holiday %>%
  group_by(days_from_holiday) %>%
  summarise(
    n = n(),
    avg_arr_delay = mean(arr_delay, na.rm = TRUE),
    pct_delayed_15 = mean(delayed_15, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(days_from_holiday)

print(delay_by_relative_day)

# Plots ---------------

# Average arrival delay around holidays

ggplot(delay_by_relative_day, aes(x = days_from_holiday, y = avg_arr_delay)) +
  geom_line(linewidth = 1.1, color = "#1f4e79") +
  geom_point(size = 3.2, color = "#1f4e79") +
  geom_text(
    aes(label = round(avg_arr_delay, 1)),
    vjust = -0.8,
    size = 3.5,
    color = "gray25"
  ) +
  scale_x_continuous(breaks = delay_by_relative_day$days_from_holiday) +
  labs(
    title = "Average Arrival Delay Around Federal Holidays",
    subtitle = "Average arrival delay for flights within the holiday analysis window",
    x = "Days Relative to Holiday",
    y = "Average Arrival Delay (minutes)",
    caption = "Negative values indicate days before the holiday; positive values indicate days after"
  ) +
  report_theme

# Probability of 15+ minute delay around holidays

ggplot(delay_by_relative_day, aes(x = days_from_holiday, y = pct_delayed_15)) +
  geom_line(linewidth = 1.1, color = "#8b1e3f") +
  geom_point(size = 3.2, color = "#8b1e3f") +
  
  geom_text(
    aes(
      label = scales::percent(pct_delayed_15, accuracy = 0.1),
      vjust = ifelse(pct_delayed_15 == min(pct_delayed_15), -1.4, -1.8),
      hjust = ifelse(days_from_holiday == -2, 0, 0.5)
    ),
    size = 3.6,
    color = "gray20"
  ) +
  
  scale_x_continuous(breaks = delay_by_relative_day$days_from_holiday) +
  
  scale_y_continuous(
    limits = c(0.1, 0.25),
    breaks = seq(0.1, 0.25, by = 0.05),
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  
  labs(
    title = "Share of Flights Delayed 15+ Minutes Around Holidays",
    subtitle = "Delay share rises after the holiday date in this sample",
    x = "Days Relative to Holiday",
    y = "Flights Delayed 15+ Minutes",
    caption = "Delay defined as arrival delay of at least 15 minutes"
  ) +
  
  report_theme

# Average arrival delay by holiday

delay_by_holiday_plot <- delay_by_holiday %>%
  mutate(label_hjust = ifelse(avg_arr_delay >= 0, -0.15, 1.15))

ggplot(delay_by_holiday_plot,
       aes(x = reorder(holiday_name, avg_arr_delay), y = avg_arr_delay)) +
  geom_col(fill = "#2f6690", width = 0.72) +
  geom_text(
    aes(label = round(avg_arr_delay, 1), hjust = label_hjust),
    size = 3.6,
    color = "gray20"
  ) +
  scale_y_continuous(
    limits = c(-2, 15),
    breaks = c(0, 5, 10, 15),
    expand = expansion(mult = c(0, 0.02))
  ) +
  coord_flip() +
  labs(
    title = "Average Arrival Delay by Holiday",
    subtitle = "Each bar reflects a 7-day window: 3 days before, the holiday itself, and 3 days after",
    x = NULL,
    y = "Average Arrival Delay (minutes)",
    caption = "Holiday-level metrics are based on a ±3 day window around each federal holiday, not the holiday date alone"
  ) +
  report_theme +
  theme(
    panel.grid.major.y = element_blank()
  )

# Dataframe -----------------

# Make sure outcome is numeric 0/1
flights_holiday <- flights_holiday %>%
  mutate(
    delayed_15 = as.integer(arr_del15),
    prev_arr_late_15 = as.factor(prev_arr_late_15)
  )

# Keep only rows with complete data for model columns
model_df <- flights_holiday %>%
  select(
    delayed_15,
    days_from_holiday,
    dep_hour_local,
    distance,
    dep_temp_c,
    dep_wind_speed_m_s,
    dep_ceiling_height_m,
    prev_arr_delay,
    prev_dep_delay,
    turnaround_minutes,
    prev_arr_late_15
  ) %>%
  filter(complete.cases(.))

# Turnaround Analysis ----------

# Bin turnaround into interpretable buckets
model_df <- model_df %>%
  mutate(
    turnaround_bucket = case_when(
      turnaround_minutes < 30 ~ "<30",
      turnaround_minutes < 60 ~ "30-60",
      turnaround_minutes < 90 ~ "60-90",
      turnaround_minutes < 120 ~ "90-120",
      TRUE ~ "120+"
    ),
    turnaround_bucket = factor(
      turnaround_bucket,
      levels = c("<30", "30-60", "60-90", "90-120", "120+")
    )
  )

turnaround_summary <- model_df %>%
  group_by(turnaround_bucket) %>%
  summarise(
    pct_delayed = mean(delayed_15, na.rm = TRUE),
    avg_prev_arr_delay = mean(prev_arr_delay, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  arrange(turnaround_bucket)

ggplot(turnaround_summary, aes(x = turnaround_bucket, y = pct_delayed)) +
  geom_col(fill = "steelblue", width = 0.72) +
  geom_text(
    aes(label = scales::percent(pct_delayed, accuracy = 0.1)),
    vjust = -0.5,
    size = 3.6,
    color = "gray20"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "Delay Rate by Turnaround Time",
    x = "Turnaround Time (minutes)",
    y = "Probability of Delay"
  ) +
  report_theme

# Previous Delay Effect -------------

previous_delay_plot_df <- model_df %>%
  filter(
    !is.na(prev_arr_delay),
    !is.na(delayed_15),
    prev_arr_delay >= 0,
    prev_arr_delay <= 1440
  ) %>%
  mutate(
    prev_arr_delay_hours = prev_arr_delay / 60
  )

ggplot(previous_delay_plot_df, aes(x = prev_arr_delay_hours, y = delayed_15)) +
  geom_smooth(
    method = "gam",
    formula = y ~ s(x, bs = "cs"),
    se = FALSE,
    linewidth = 1.2,
    color = "#2f66ff"
  ) +
  scale_x_continuous(
    limits = c(0, 24),
    breaks = seq(0, 24, by = 4)
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2)
  ) +
  labs(
    title = "Effect of Previous Arrival Delay on Current Delay",
    x = "Previous Arrival Delay (hours)",
    y = "Probability of Current Delay",
    caption = "Delay defined as arrival delay of 15+ minutes"
  ) +
  report_theme

# Weather -----

# Wind speed bins

model_df <- model_df %>%
  mutate(
    dep_wind_speed_mph = dep_wind_speed_m_s * 2.23694,
    wind_bucket = cut(
      dep_wind_speed_mph,
      breaks = c(0, 10, 20, 30, 40, Inf),
      labels = c("0–10", "10–20", "20–30", "30–40", "40+"),
      include.lowest = TRUE
    )
  )

wind_summary <- model_df %>%
  group_by(wind_bucket) %>%
  summarise(
    pct_delayed = mean(delayed_15, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

ggplot(wind_summary, aes(x = wind_bucket, y = pct_delayed)) +
  geom_col(fill = "#a23b2a", width = 0.72) +
  geom_text(
    aes(label = scales::percent(pct_delayed, accuracy = 0.1)),
    vjust = -0.5,
    size = 3.5,
    color = "gray20"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "Delay Rate by Departure Wind Speed",
    subtitle = "Higher wind speeds are associated with somewhat higher delay rates",
    x = "Wind Speed at Departure (mph)",
    y = "Delay Rate",
    caption = "Delay defined as arrival delay of at least 15 minutes"
  ) +
  report_theme

# Ceiling height buckets

model_df <- model_df %>%
  mutate(
    dep_ceiling_height_ft = dep_ceiling_height_m * 3.28084,
    ceiling_bucket = case_when(
      dep_ceiling_height_ft < 1500  ~ "<1,500",
      dep_ceiling_height_ft < 3000  ~ "1,500–2,999",
      dep_ceiling_height_ft < 6000  ~ "3,000–5,999",
      dep_ceiling_height_ft < 10000 ~ "6,000–9,999",
      dep_ceiling_height_ft < 16000 ~ "10,000–15,999",
      TRUE                          ~ "16,000+"
    )
  )

ceiling_summary <- model_df %>%
  group_by(ceiling_bucket) %>%
  summarise(
    pct_delayed = mean(delayed_15, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    ceiling_bucket = factor(
      ceiling_bucket,
      levels = c("<1,500", "1,500–2,999", "3,000–5,999",
                 "6,000–9,999", "10,000–15,999", "16,000+")
    )
  )

ggplot(ceiling_summary, aes(x = ceiling_bucket, y = pct_delayed)) +
  geom_col(fill = "#4c956c", width = 0.72) +
  geom_text(
    aes(label = scales::percent(pct_delayed, accuracy = 0.1)),
    vjust = -0.5,
    size = 3.5,
    color = "gray20"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "Delay Rate by Ceiling Height",
    subtitle = "Lower ceiling heights generally indicate less favorable operating conditions",
    x = "Ceiling Height at Departure (feet)",
    y = "Delay Rate"
  ) +
  report_theme

# GLM ------------

# Sample if still large
set.seed(123)
if (nrow(model_df) > 50000) {
  model_df <- model_df %>% slice_sample(n = 50000)
}

# Logistic regression
holiday_glm_light <- glm(
  delayed_15 ~ days_from_holiday +
    dep_hour_local +
    distance +
    dep_temp_c +
    dep_wind_speed_m_s +
    dep_ceiling_height_m +
    prev_arr_delay +
    prev_dep_delay +
    turnaround_minutes +
    prev_arr_late_15,
  data = model_df,
  family = binomial()
)

summary(holiday_glm_light)

# Decision Tree --------------------

set.seed(123)

model_df_tree <- model_df %>%
  rename(
    "Days from Holiday" = days_from_holiday,
    "Dep Hour" = dep_hour_local,
    "Distance" = distance,
    "Temp" = dep_temp_c,
    "Wind Speed" = dep_wind_speed_m_s,
    "Ceiling Height" = dep_ceiling_height_m,
    "Previous Arrival Delay" = prev_arr_delay,
    "Previous Departure Delay" = prev_dep_delay,
    "Turnaround Time" = turnaround_minutes,
    "Previous Delay (15+)" = prev_arr_late_15,
    "Delayed (15+)" = delayed_15
  )

tree_model <- rpart(
  `Delayed (15+)` ~ 
    `Days from Holiday` +
    `Dep Hour` +
    `Distance` +
    `Temp` +
    `Wind Speed` +
    `Ceiling Height` +
    `Previous Arrival Delay` +
    `Previous Departure Delay` +
    `Turnaround Time` +
    `Previous Delay (15+)`,
  data = model_df_tree,
  method = "class",
  control = rpart.control(cp = 0.001, maxdepth = 5)
)

# Plot
rpart.plot(
  tree_model,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  split.cex = 0.81,
  tweak = 1.2
)

# Random Forest ------------------

set.seed(123)

rf_model <- randomForest(
  as.factor(delayed_15) ~ days_from_holiday +
    dep_hour_local +
    distance +
    dep_temp_c +
    dep_wind_speed_m_s +
    dep_ceiling_height_m +
    prev_arr_delay +
    prev_dep_delay +
    turnaround_minutes +
    prev_arr_late_15,
  data = model_df,
  ntree = 100,
  mtry = 3,
  importance = TRUE,
  sampsize = c(5000, 5000)
)

print(rf_model)

# Predicted probabilities
rf_probs <- predict(rf_model, type = "prob")[, 2]

# ROC object and AUC
roc_curve <- pROC::roc(model_df$delayed_15, rf_probs)
rf_auc <- pROC::auc(roc_curve)

# Variable importance dataframe

importance_df <- as.data.frame(rf_model$importance) %>%
  tibble::rownames_to_column("variable") %>%
  arrange(desc(MeanDecreaseAccuracy))

# Clean variable labels for plotting/reporting
importance_df <- importance_df %>%
  mutate(
    variable_clean = dplyr::recode(
      variable,
      days_from_holiday = "Days from Holiday",
      dep_hour_local = "Departure Hour",
      distance = "Distance",
      dep_temp_c = "Departure Temperature",
      dep_wind_speed_m_s = "Wind Speed",
      dep_ceiling_height_m = "Ceiling Height",
      prev_arr_delay = "Previous Arrival Delay",
      prev_dep_delay = "Previous Departure Delay",
      turnaround_minutes = "Turnaround Time",
      prev_arr_late_15 = "Previous Arrival 15+ Delay Flag"
    )
  )

print(importance_df)

# Feature importance plot

ggplot(
  importance_df,
  aes(
    x = reorder(variable_clean, MeanDecreaseAccuracy),
    y = MeanDecreaseAccuracy
  )
) +
  geom_col(fill = "#2f6690", width = 0.72) +
  geom_text(
    aes(label = round(MeanDecreaseAccuracy, 1)),
    hjust = -0.15,
    size = 3.5,
    color = "gray20"
  ) +
  coord_flip() +
  expand_limits(y = max(importance_df$MeanDecreaseAccuracy) * 1.12) +
  labs(
    title = "Feature Importance (Random Forest)",
    subtitle = "Measured by mean decrease in accuracy",
    x = NULL,
    y = "Importance"
  ) +
  report_theme +
  theme(
    panel.grid.major.y = element_blank()
  )

# ROC curve dataframe

roc_df <- data.frame(
  specificity = roc_curve$specificities,
  sensitivity = roc_curve$sensitivities
) %>%
  mutate(
    false_positive_rate = 1 - specificity,
    true_positive_rate = sensitivity
  )

# ROC curve plot

ggplot(roc_df, aes(x = false_positive_rate, y = true_positive_rate)) +
  geom_line(color = "#8b1e3f", linewidth = 1.2) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "gray50"
  ) +
  annotate(
    "label",
    x = 0.68,
    y = 0.15,
    label = paste0("AUC = ", round(rf_auc, 3)),
    size = 4,
    fill = "white",
    color = "black",
    label.size = 0.25
  ) +
  labs(
    title = "ROC Curve (Random Forest)",
    subtitle = "Model discrimination performance across classification thresholds",
    x = "False Positive Rate",
    y = "True Positive Rate",
    caption = "Dashed line represents random-chance performance"
  ) +
  coord_equal() +
  report_theme

# XGBoost -----------

# Make sure target is numeric 0/1
model_df <- model_df %>%
  mutate(
    delayed_15 = as.integer(as.character(delayed_15))
  )

# Build predictor matrix
X <- model.matrix(
  ~ days_from_holiday +
    dep_hour_local +
    distance +
    dep_temp_c +
    dep_wind_speed_m_s +
    dep_ceiling_height_m +
    prev_arr_delay +
    prev_dep_delay +
    turnaround_minutes +
    prev_arr_late_15,
  data = model_df
)[, -1]

# Label
y <- model_df$delayed_15

# DMatrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Train model
xgb_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1
  ),
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

# Importance
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)
xgb.plot.importance(importance_matrix)
# Clustering ------------
cluster_df <- model_df %>%
  select(
    dep_hour_local,
    dep_wind_speed_m_s,
    dep_temp_c,
    prev_arr_delay,
    turnaround_minutes
  ) %>%
  na.omit() %>%
  filter(
    turnaround_minutes > 0,
    turnaround_minutes < 600,
    abs(prev_arr_delay) < 300
  )

# Scale variables and run k-means
cluster_scaled <- scale(cluster_df)

set.seed(123)
kmeans_model <- kmeans(cluster_scaled, centers = 4)

cluster_df$cluster <- as.factor(kmeans_model$cluster)

# Rename clusters for readability
cluster_df$cluster_label <- recode(
  cluster_df$cluster,
  "1" = "Long Turnaround / Low Delay",
  "2" = "High Delay Risk",
  "3" = "Short Turnaround / Recovery",
  "4" = "Moderate Operations"
)

# Compute cluster centers for plotting
centers <- cluster_df %>%
  group_by(cluster_label) %>%
  summarise(
    turnaround_minutes = mean(turnaround_minutes),
    prev_arr_delay = mean(prev_arr_delay),
    .groups = "drop"
  )

# Plot
ggplot(cluster_df, aes(x = turnaround_minutes, y = prev_arr_delay, color = cluster_label)) +
  geom_point(alpha = 0.15, size = 1) +
  geom_point(
    data = centers,
    aes(x = turnaround_minutes, y = prev_arr_delay),
    color = "black",
    size = 5,
    shape = 4,
    stroke = 2,
    inherit.aes = FALSE
  ) +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Flight Clusters: Turnaround vs Previous Delay",
    x = "Turnaround Minutes",
    y = "Previous Arrival Delay",
    color = "Cluster Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top",
    panel.grid.minor = element_blank(),
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10)
  ) +
  guides(color = guide_legend(nrow = 2))

# Cluster summary table
cluster_summary <- cluster_df %>%
  group_by(cluster, cluster_label) %>%
  summarise(
    dep_hour_local = mean(dep_hour_local),
    dep_wind_speed_m_s = mean(dep_wind_speed_m_s),
    dep_temp_c = mean(dep_temp_c),
    prev_arr_delay = mean(prev_arr_delay),
    turnaround_minutes = mean(turnaround_minutes),
    n = n(),
    .groups = "drop"
  )

print(cluster_summary)
# AIRPORT DELAY BUBBLE MAP --------

# Create legend breaks based on the actual plotted data
size_breaks <- c(
  min(airport_map_data$delayed_flights, na.rm = TRUE),
  median(airport_map_data$delayed_flights, na.rm = TRUE),
  max(airport_map_data$delayed_flights, na.rm = TRUE)
) %>%
  round(0) %>%
  unique()

# Build top 5 annotation text with full airport names
top5_text <- paste0(
  "Top 5 Airports by Delay Rate",
  "\n(min. 50 flights)\n\n",
  paste0(
    seq_len(nrow(top5_airports)), ". ",
    top5_airports$name, " (", top5_airports$dest, ")  ",
    percent(top5_airports$delay_rate, accuracy = 1),
    collapse = "\n"
  )
)

# Plot bubble map
ggplot() +
  geom_polygon(
    data = us_map,
    aes(x = long, y = lat, group = group),
    fill = "gray94",
    color = "gray75",
    linewidth = 0.25
  ) +
  geom_point(
    data = airport_map_data,
    aes(
      x = lon,
      y = lat,
      size = delayed_flights,
      color = delay_rate
    ),
    alpha = 0.65
  ) +
  geom_label_repel(
    data = top5_airports,
    aes(
      x = lon,
      y = lat,
      label = dest
    ),
    size = 3.5,
    box.padding = 0.25,
    point.padding = 0.15,
    segment.color = "gray40",
    min.segment.length = 0,
    max.overlaps = Inf
  ) +
  annotate(
    "label",
    x = -110,   # move right edge inside plot
    y = 18,    # move up into visible area
    hjust = 1, # anchor right side
    vjust = 0, # anchor bottom
    label = top5_text,
    size = 3.2,
    label.size = 0.3,
    fill = "white",
    color = "black",
    lineheight = 1.05
  ) +
  scale_color_gradient(
    low = "seagreen3",
    high = "red2",
    labels = percent_format(accuracy = 1),
    name = "Delay Rate"
  ) +
  scale_size_continuous(
    range = c(2, 10),
    breaks = c(100, 1000, 5000, 10000),
    labels = scales::comma_format(),
    name = "Delayed Flights"
  ) +
  coord_fixed(
    1.25,
    xlim = c(-125, -66),
    ylim = c(24, 50),
    clip = "off"
  ) +
  labs(
    title = "US Airport Arrival Delays During the Holiday Window",
    subtitle = "Bubble size shows delayed flight count; color shows destination-airport delay rate",
    caption = "Delay defined as arrival delay of 15+ minutes"
  ) +
  theme_void(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 22, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, margin = ggplot2::margin(b = 8)),
    plot.caption = element_text(size = 10, color = "gray30", hjust = 0.5),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    plot.margin = ggplot2::margin(10, 10, 10, 10)
  )

# Zoomed map: BWI / EWR / JFK REGION

zoom_ne <- ggplot() +
  geom_polygon(
    data = us_map,
    aes(x = long, y = lat, group = group),
    fill = "gray94",
    color = "gray75",
    linewidth = 0.25
  ) +
  geom_point(
    data = airport_map_data,
    aes(
      x = lon,
      y = lat,
      size = delayed_flights,
      color = delay_rate
    ),
    alpha = 0.75
  ) +
  
  # Highlight key airports
  geom_point(
    data = airport_map_data %>% filter(dest %in% c("BWI", "EWR", "JFK")),
    aes(x = lon, y = lat),
    color = "black",
    size = 5,
    shape = 1,
    stroke = 1.5
  ) +
  
  geom_label_repel(
    data = airport_map_data %>% filter(dest %in% c("BWI", "EWR", "JFK")),
    aes(x = lon, y = lat, label = dest),
    size = 4.5,
    box.padding = 0.4,
    point.padding = 0.3,
    segment.color = "gray30"
  ) +
  
  # Zoom into Northeast corridor
  coord_fixed(
    1.3,
    xlim = c(-80, -70),
    ylim = c(37, 43)
  ) +
  
  scale_color_gradient(
    low = "seagreen3",
    high = "red2",
    labels = scales::percent_format(accuracy = 1),
    name = "Delay Rate"
  ) +
  
  scale_size_continuous(range = c(2, 10), guide = "none") +
  
  labs(
    title = "Zoomed View: BWI – EWR – JFK Corridor",
    subtitle = "High-density air traffic region with overlapping delay patterns"
  ) +
  
  theme_void() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5)
  )

zoom_ne

# Hour of Day Plot ------------
delay_by_hour <- model_df %>%
  group_by(dep_hour_local) %>%
  summarise(
    pct_delayed = mean(delayed_15, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

delay_by_hour_filtered <- delay_by_hour %>%
  filter(n >= 30)

# Start plot at 4 AM to focus on same-day delay buildup
delay_by_hour_plot <- delay_by_hour_filtered %>%
  filter(dep_hour_local >= 4)

ggplot(delay_by_hour_plot, aes(x = dep_hour_local, y = pct_delayed)) +
  geom_line(linewidth = 1.2, color = "#3b3b98") +
  geom_point(size = 3.2, color = "#3b3b98") +
  geom_text(
    aes(label = scales::percent(pct_delayed, accuracy = 1)),
    vjust = -0.8,
    size = 3.5,
    color = "gray20"
  ) +
  scale_x_continuous(
    breaks = 4:23
  ) +
  scale_y_continuous(
    limits = c(0, 0.50),
    breaks = seq(0, 0.50, by = 0.10),
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Delay Rate by Departure Hour",
    subtitle = "Flights before 4 AM excluded to emphasize how delays build through the operating day",
    x = "Departure Hour (Local Time)",
    y = "Delay Rate",
    caption = "Hours with fewer than 30 flights removed; overnight departures excluded from this view"
  ) +
  report_theme