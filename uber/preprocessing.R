library(tidyverse)
library(lubridate)
library(riem)

all_data_list = vector("list", 8)
for (i in 1:8){
  all_data_list[[i]] = vector("list", length(file_names[[i]]))
  for (j in 1:length(file_names[[i]])){
    all_data_list[[i]][[j]] = read_csv(paste0("data/", file_names[[i]][[j]])) %>%
      mutate(Date = mdy(Date)) %>% arrange(Date)
  }
}
all_data_list_small = lapply(all_data_list, function(x) do.call(rbind, x))
all_data = do.call(rbind, all_data_list_small) %>% arrange(Date)
# check if all data are indeed downloaded
for (i in 1:length(all_data_list)){
  for (j in 1:length(all_data_list[[i]])){
    temp = all_data_list[[i]][[j]]

    logic1 = as.Date(temp$Date[1]) == as.Date(sd_dates[i])
    logic2 = as.Date(temp$Date[nrow(temp)]) == as.Date(ed_dates[i])
    logic3 = temp$`Origin Movement ID`[1] == edges[j, 1]
    logic4 = temp$`Destination Movement ID`[1] == edges[j, 2]

    if (!(logic1 & logic2 & logic3 & logic4)) cat("i", i, "j", j, "\n")
  }
}

###### missing data
thin_data = all_data %>% rename(origin = `Origin Movement ID`,
                                destination = `Destination Movement ID`,
                                Daily = `Daily Mean Travel Time (Seconds)`,
                                AM = `AM Mean Travel Time (Seconds)`,
                                PM = `PM Mean Travel Time (Seconds)`,
                                Midday = `Midday Mean Travel Time (Seconds)`,
                                Evening = `Evening Mean Travel Time (Seconds)`,
                                EarlyMorning = `Early Morning Mean Travel Time (Seconds)`) %>%
  select(Date, origin, destination, AM, PM, Midday, Evening, EarlyMorning)
avg_time = thin_data %>% pivot_longer(cols = c("AM", "PM", "Midday", "Evening", "EarlyMorning"),
                                      names_to = "Period", values_to = "Time") %>%
  group_by(origin, destination, Period) %>% summarise(mean_time = mean(Time, na.rm = TRUE))
avg_time_list = vector("list", nrow(edges))
for (i in 1:nrow(edges)){
  avg_time_list[[i]] = avg_time %>% filter(origin == edges[i, 1], destination == edges[i, 2])
}
avg_time = do.call(bind_rows, avg_time_list)
# using average traveling time to impute missing observations
for (i in 1:nrow(avg_time)){
  temp = avg_time[i, ]
  missing_entries = thin_data %>% filter(origin == temp$origin,
                                         destination == temp$destination) %>%
    select(temp$Period) %>% is.na()
  thin_data[(thin_data$origin == temp$origin) & (thin_data$destination == temp$destination), temp$Period][missing_entries, ] =
    temp$mean_time
}
sum(is.na(thin_data))
thin_data = thin_data %>% pivot_longer(cols = c("AM", "PM", "Midday", "Evening", "EarlyMorning"),
                                       names_to = "Period", values_to = "Time")

#### weather features
# https://cran.r-project.org/web/packages/riem/vignettes/riem_package.html
networks = riem_networks()
networks[grep("California", networks$name), ]
stations = riem_stations("CA_ASOS")
stations[grep("LAX", stations$id), ]
# measures <- riem_measures(station = "LAX", date_start = '2017-01-01', date_end = '2020-01-01')
# write_csv(measures, "weather_measures.csv")
measures = read_csv("weather_measures.csv")
AM = (8 <= hour(measures$valid)) & (hour(measures$valid) <= 10)
Midday = (11 <= hour(measures$valid)) & (hour(measures$valid) <= 16)
PM = (17 <= hour(measures$valid)) & (hour(measures$valid) <= 19)
Evening = ((20 <= hour(measures$valid)) & (hour(measures$valid) <= 23)) | (hour(measures$valid) == 0)
EarlyMorning = (1 <= hour(measures$valid)) & (hour(measures$valid) <= 7)
measures$Period = NA
measures[AM, "Period"] = "AM"
measures[Midday, "Period"] = "Midday"
measures[PM, "Period"] = "PM"
measures[Evening, "Period"] = "Evening"
measures[EarlyMorning, "Period"] = "EarlyMorning"
measures$Date = as.Date(measures$valid)
measures_summary = measures %>% select(Date, Period, tmpf, sknt, p01i, vsby) %>%
  group_by(Date, Period) %>%
  summarise(Temp = mean(tmpf, na.rm = T),
            WindSpeed = mean(sknt, na.rm = T),
            Rain = mean(p01i, na.rm = T),
            Visibility = mean(vsby, na.rm = T))

### calendar features
measures_summary$Date = ymd(measures_summary$Date)
measures_summary$weekday = wday(measures_summary$Date)
measures_summary$month = month(measures_summary$Date)
common_X = measures_summary

### lagging traveling times features
create_lags <- function(avg_time, thin_data, lag_seq = c(1, 7)){
  lag_list = vector("list", nrow(avg_time))
  for (i in 1:nrow(avg_time)){
    index = avg_time[i, ]
    temp = thin_data %>% filter(origin == index$origin,
                                destination == index$destination, Period == index$Period)

    stopifnot(sum(temp$Date != c(temp %>% arrange(Date) %>% select(Date))[[1]])==0)
    lags = lapply(lag_seq, function(x) lag(temp$Time, n = x))
    lags = do.call(cbind, lags)
    colnames(lags) = paste0("lag", lag_seq)
    lags = as_tibble(lags)
    lag_list[[i]] = bind_cols(temp, lags)
  }
  lag_list
}
thin_data_list = create_lags(avg_time, thin_data, lag_seq = c(1, 7))
thin_data_list_agg = vector("list", nrow(edges))
for (i in 1:nrow(edges)){
  index = ((i-1)*5 + 1):(i*5)
  thin_data_list_agg[[i]] = do.call(rbind, thin_data_list[index]) %>% arrange(Date, Period)
}
for (i in 1:nrow(edges)){
  thin_data_list_agg[[i]] = thin_data_list_agg[[i]][complete.cases(thin_data_list_agg[[i]]), ]
}
outcomes_list = vector("list", nrow(edges))
for (i in 1:nrow(edges)){
  origin = thin_data_list_agg[[i]]$origin[1]
  destination = thin_data_list_agg[[i]]$destination[1]
  stopifnot({
    origin == edges[i, 1]
    destination == edges[i, 2]
  })
  temp = thin_data_list_agg[[i]] %>% select(Time, starts_with("lag"))
  colnames(temp) =
    c(paste0("Y", "_origin", origin, "_destination", destination),
      paste0("lag", c(1, 7), "_origin", origin, "_destination", destination))
  outcomes_list[[i]] = temp
}
identifiers = thin_data_list_agg[[1]] %>% select(Date, origin, destination, Period)
outcomes = do.call(bind_cols, outcomes_list)
X_specific = outcomes %>% select(starts_with("lag"))
X_common = identifiers %>% select(Date, Period) %>%  left_join(common_X, by = c("Date", "Period"))
X = bind_cols(X_common, X_specific)
Y = outcomes %>% select(starts_with("Y"))

### create datasets
X %>% filter((year(X$Date) == 2019)&(month(X$Date) >= 7)) %>%
  select(c("Period", "Temp", "WindSpeed",
           "Rain", "Visibility", "weekday", "month"),
         starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_halfyear.csv")
Y %>% filter((year(X$Date) == 2019)&(month(X$Date) >= 7)) %>%
  write_csv("Y_halfyear.csv")

X %>% filter(year(X$Date) == 2019) %>%
  select(c("Period", "Temp", "WindSpeed",
           "Rain", "Visibility", "weekday", "month"),
         starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_oneyear.csv")
Y %>% filter(year(X$Date) == 2019) %>% write_csv("Y_oneyear.csv")

X %>% filter((year(X$Date) == 2019) | ( (year(X$Date) == 2018) & (month(X$Date) >= 7) )) %>%
  select(c("Period", "Temp", "WindSpeed",
           "Rain", "Visibility", "weekday", "month"),
         starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_onehalfyear.csv")
Y %>% filter((year(X$Date) == 2019) | ( (year(X$Date) == 2018) & (month(X$Date) >= 7) )) %>%
  write_csv("Y_onehalfyear.csv")

X %>% filter((year(X$Date) == 2019) | (year(X$Date) == 2018)) %>%
  select(c("Period", "Temp", "WindSpeed",
           "Rain", "Visibility", "weekday", "month"),
         starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_twoyear.csv")
Y %>% filter((year(X$Date) == 2019) | (year(X$Date) == 2018)) %>% write_csv("Y_twoyear.csv")

