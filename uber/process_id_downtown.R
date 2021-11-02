library(tidyverse)
library(lubridate)
library(riem)

########### downloading
make_small_matrix <- function(x, y){
  XX = rep(x, length(y))
  YY = y
  temp = cbind(XX, YY)
  # temp2 = matrix(0, nrow = nrow(temp), ncol = 2)
  # temp2[, 1] = temp[, 2]
  # temp2[, 2] = temp[, 1]
  # rbind(temp, temp2)
  temp
}

edges = list()
edges[["1221"]] = make_small_matrix(1221, c(1222, 1220))
edges[["1222"]] = make_small_matrix(1222, c(1220))
edges[["1220"]] = make_small_matrix(1220 , c(1230, 1223, 1224, 1390))
edges[["1230"]] = make_small_matrix(1230 , c(1223, 1229, 1228, 1232, 1235))
edges[["1223"]] = make_small_matrix(1223 , c(1224, 1229))
edges[["1224"]] = make_small_matrix(1224 , c(1390, 1229))
edges[["1390"]] = make_small_matrix(1390 , c(1228, 1234, 1380))
edges[["1229"]] = make_small_matrix(1229 , c(1228))
edges[["1228"]] = make_small_matrix(1228 , c(1234, 1232, 1233))
edges[["1234"]] = make_small_matrix(1234 , c(1380, 1233))
edges[["1380"]] = make_small_matrix(1380 , c(1382))
edges[["1232"]] = make_small_matrix(1232 , c(1233, 1254))
edges[["1233"]] = make_small_matrix(1233 , c(1380, 1254, 1255, 1263))

edges[["1235"]] = make_small_matrix(1235 , c(1254, 1237))
edges[["1254"]] = make_small_matrix(1254 , c(1255, 1252, 1251))
edges[["1255"]] = make_small_matrix(1255 , c(1263, 1258))
edges[["1263"]] = make_small_matrix(1263 , c(1382, 1260, 1262))
edges[["1382"]] = make_small_matrix(1382 , c(1384))

edges[["1237"]] = make_small_matrix(1237 , c(1252, 1236, 1239))
edges[["1252"]] = make_small_matrix(1252 , c(1251, 1253))
edges[["1251"]] = make_small_matrix(1251 , c(1255, 1250, 1248))
edges[["1236"]] = make_small_matrix(1236 , c(1253, 1238))
edges[["1253"]] = make_small_matrix(1253 , c(1250, 1251))

edges[["1239"]] = make_small_matrix(1239 , c(1238, 1240))
edges[["1238"]] = make_small_matrix(1238 , c(1250, 1249, 1241))
edges[["1250"]] = make_small_matrix(1250 , c(1248, 1249))
edges[["1249"]] = make_small_matrix(1249 , c(1257, 1246))
edges[["1248"]] = make_small_matrix(1248 , c(1258, 1249))
edges[["1258"]] = make_small_matrix(1258 , c(1260, 1257))
edges[["1257"]] = make_small_matrix(1257 , c(1260, 1256))
edges[["1260"]] = make_small_matrix(1260 , c(1262, 1259))
edges[["1262"]] = make_small_matrix(1262 , c(1384, 1261))
edges[["1384"]] = make_small_matrix(1384 , c(1383))

edges[["1240"]] = make_small_matrix(1240 , c(1241, 1243))
edges[["1241"]] = make_small_matrix(1241 , c(1246, 1247, 1243))
edges[["1246"]] = make_small_matrix(1246 , c(1256, 1247))
edges[["1247"]] = make_small_matrix(1247 , c(1256, 1245))
edges[["1256"]] = make_small_matrix(1256 , c(1259))
edges[["1259"]] = make_small_matrix(1259 , c(1261))
edges[["1261"]] = make_small_matrix(1261 , c(1383))

edges[["1243"]] = make_small_matrix(1243 , c(1245, 1244, 1242))
edges[["1245"]] = make_small_matrix(1245 , c(1256, 1244))
edges[["1242"]] = make_small_matrix(1242 , c(1244))
edges = do.call(rbind, edges)

unique_edges = unique(c(edges))
census = rep(0, length(unique_edges))
names(census) = unique_edges
census["1221"] = 206032
census["1222"] = 206050
census["1220"] = 206031
census["1230"] = 207400
census["1223"] = 206200
census["1224"] = 206300
census["1390"] = 226002
census["1229"] = 207302
census["1228"] = 207301
census["1234"] = 207900
census["1380"] = 224010
census["1232"] = 207502
census["1233"] = 207710
census["1235"] = 208000
census["1254"] = 209200
census["1255"] = 209300
census["1263"] = 210010
census["1382"] = 224200
census["1237"] = 208302
census["1252"] = 209103
census["1251"] = 209102
census["1236"] = 208301
census["1253"] = 209104
census["1239"] = 208402
census["1238"] = 208401
census["1250"] = 208904
census["1249"] = 208903
census["1248"] = 208902
census["1258"] = 209403
census["1257"] = 209402
census["1260"] = 209520
census["1262"] = 209820
census["1384"] = 224320
census["1240"] = 208501
census["1241"] = 208502
census["1246"] = 208801
census["1247"] = 208802
census["1256"] = 209401
census["1259"] = 209510
census["1261"] = 209810
census["1243"] = 208620
census["1245"] = 208720
census["1242"] = 208610
census["1383"] = 224310
census["1244"] = 208710

# unique_edges

city="los_angeles"
sd_dates = c('2019-01-01', '2019-04-01', '2019-07-01', '2019-10-01',
             "2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01",
             "2017-01-01", "2017-04-01", "2017-07-01", "2017-10-01")
ed_dates = c('2019-03-31', '2019-06-30', '2019-09-30', '2019-12-31',
             '2018-03-31', '2018-06-30', '2018-09-30', '2018-12-31',
             '2017-03-31', '2017-06-30', '2017-09-30', '2017-12-31')
coordinates = "&lat.=33.9978718&lng.=-118.4798331&z.=11.96"

links = vector("list", (length(sd_dates)))
for (i in 1:(length(sd_dates))){
  sd = sd_dates[i]
  ed = ed_dates[i]
  links[[i]] = vector("list", nrow(edges))
  for (j in 1:nrow(edges)){
    origin_code = edges[j, 1]
    des_code = edges[j, 2]

    links[[i]][[j]] = paste0('https://movement.uber.com/explore/', city,
                             '/travel-times/query?si=', origin_code,
                             '&ti=', des_code,
                             '&ag=censustracts&dt[tpb]=ALL_DAY&dt[wd;]=1,2,3,4,5,6,7&dt[dr][sd]=', sd,
                             '&dt[dr][ed]=', ed, '&cd=&sa;=&sdn=&ta;=&tdn=', coordinates, '&lang=en-US')
  }
}
file_names = vector("list", (length(sd_dates)))
for (i in 1:(length(sd_dates))){
  sd = sd_dates[i]
  ed = ed_dates[i]
  file_names[[i]] = vector("list", nrow(edges))
  for (j in 1:nrow(edges)){
    origin_code = edges[j, 1]
    des_code = edges[j, 2]

    file_names[[i]][[j]] = paste0("origin", origin_code,
                                  "_des", des_code,
                                  "_sd", sd, "_ed", ed, ".csv")
  }
}

i = 12
j = 93
origin_code =edges[j, 1]
des_code = edges[j, 2]
sd = sd_dates[i]
ed = ed_dates[i]
cat("i ", i, "; j ", j)
cat("origin id ", origin_code, "; destination id ", des_code)
cat("origin census ", census[as.character(origin_code)],
    "; destination census", census[as.character(des_code)])
cat("starting date ", sd, "; ending date ", ed)
browseURL(links[[i]][[j]])

file.rename("~/Downloads/Travel_Times_Daily.csv", paste0("~/Downloads/", file_names[[i]][[j]]))
nrow(read_csv(paste0("~/Downloads/", file_names[[i]][[j]])))

i = 12
mistakes_id = rep(NA, nrow(edges))
mistakes_n = rep(NA, nrow(edges))
mistakes_dates = rep(NA, nrow(edges))
n = 92
for (j in 1:nrow(edges)){
  origin_code = edges[j, 1]
  des_code = edges[j, 2]
  sd = sd_dates[i]
  ed = ed_dates[i]
  temp = read_csv(paste0("~/Downloads/", file_names[[i]][[j]])) %>% arrange(Date)
  if ((mdy(temp$Date[1]) != sd)|(mdy(temp$Date[nrow(temp)]) != ed)){
    mistakes_dates[j] = T
  }
  if ((temp$`Origin Movement ID`[1]!=origin_code)|(temp$`Destination Movement ID`[1]!=des_code)){
    mistakes_id[j] = T
  }
  if (nrow(temp) != n){
    mistakes_n[j] = T
  }
}
sum(!is.na(mistakes_id))
sum(!is.na(mistakes_n))
sum(!is.na(mistakes_dates))

# which((edges[, 1]==1230)&(edges[, 2]==1229))
# name ="origin1221_des1222_sd2019-01-01_ed2019-03-31.csv"
# file.rename("~/Downloads/Travel_Times_Daily.csv", paste0("~/Downloads/", name))
# nrow(read_csv(paste0("~/Downloads/", name)))

############ preprocessing
all_data_list = vector("list", 12)
for (i in 1:12){
  all_data_list[[i]] = vector("list", length(file_names[[i]]))
  for (j in 1:length(file_names[[i]])){
    all_data_list[[i]][[j]] = read_csv(paste0("data/downtown/", file_names[[i]][[j]])) %>%
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
row_miss_freq = rowMeans(is.na(all_data))
table(row_miss_freq)/length(row_miss_freq)
miss_index = row_miss_freq > 0
miss_data = all_data[miss_index, ]
c(miss_data[1, ])
colMeans(is.na(all_data))
small_missing_data = all_data[(row_miss_freq > 0) & (row_miss_freq <= 0.15), ]
colMeans(is.na(small_missing_data))
large_missing_data = all_data[(row_miss_freq > 0.15), ]
colMeans(is.na(large_missing_data))

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

for (i in 1:nrow(avg_time)){
  temp = avg_time[i, ]
  missing_entries = thin_data %>% filter(origin == temp$origin,
                                         destination == temp$destination) %>%
    select(temp$Period) %>% is.na()
  thin_data[(thin_data$origin == temp$origin) & (thin_data$destination == temp$destination), temp$Period][missing_entries, ] =
    temp$mean_time
}
sum(is.na(thin_data))
write_csv(thin_data, "clean_data_downtown.csv")


thin_data = read_csv("clean_data_downtown.csv")
thin_data %>% group_by("Date", "origin", "destination")

thin_data = thin_data %>% pivot_longer(cols = c("AM", "PM", "Midday", "Evening", "EarlyMorning"),
                                       names_to = "Period", values_to = "Time")


###### features
####### weather
# https://cran.r-project.org/web/packages/riem/vignettes/riem_package.html
networks = riem_networks()
networks[grep("California", networks$name), ]
stations = riem_stations("CA_ASOS")
stations[grep("LAX", stations$id), ]
measures <- riem_measures(station = "LAX", date_start = '2017-01-01', date_end = '2020-01-01')
# measures = read_csv("weather_measures.csv")
# note that date_end is not included in the data
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
thin_measures = measures %>% select(Date, Period, tmpf, sknt, p01i, vsby)

colMeans(is.na(thin_measures))
na_table = is.na(thin_measures)[, 3:6]
colnames(na_table) = c("tmpf_na?", "sknt_na?", "p01i_na?", "vsby_na?")
colMeans(na_table)
# tmpf_na?    sknt_na?    p01i_na?    vsby_na?
# 0.914014125 0.005117043 0.894363257 0.004450762
thin_measures = bind_cols(thin_measures, as_tibble(na_table))
thin_measure_full = thin_measures %>% group_by(Date, Period) %>%
  summarise(tmpf_full = sum(1-`p01i_na?`),
            sknt_full = sum(1-`sknt_na?`),
            p01i_full = sum(1-`p01i_na?`),
            vsby_full = sum(1-`vsby_na?`))
colMeans(thin_measure_full == 0)
# all vars have some measurements in each time period
measures_summary = thin_measures %>% group_by(Date, Period) %>%
  summarise(Temp = mean(tmpf, na.rm = T),
            WindSpeed = mean(sknt, na.rm = T),
            Rain = mean(p01i, na.rm = T),
            Visibility = mean(vsby, na.rm = T))

# time
measures_summary$Date = ymd(measures_summary$Date)
measures_summary$weekday = wday(measures_summary$Date)
measures_summary$month = month(measures_summary$Date)
common_X = measures_summary

# thin_data = thin_data %>% left_join(measures_summary, by = c("Date", "Period"))
all_date_measure = measures_summary %>% select(Date) %>% unique() %>% arrange(Date)
all_date_correct = do.call(rbind, lapply(1:12, function(x) all_data_list[[x]][[1]])) %>%
   select(Date) %>% arrange(Date)

#
# all_date_correct_num = sapply(1:nrow(all_date_correct), function(x) mdy(all_date_correct[x, ]))
# all_date_measure_num = unlist(all_date_measure)
# for (i in 1:nrow(all_date_correct)){
#   if (!(all_date_correct_num[i] %in% all_date_measure_num)) print(i)
# }
# all_date_correct[365, ]
head(all_date_measure)
head(all_date_correct)
tail(all_date_measure)
tail(all_date_correct)

######## lagging observations
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
  # do.call(rbind, lag_list)
}
thin_data_list = create_lags(avg_time, thin_data, lag_seq = c(1, 7))
thin_data_list[1:5]
thin_data_list[6:10]
thin_data_list_agg = vector("list", nrow(edges))
for (i in 1:nrow(edges)){
  index = ((i-1)*5 + 1):(i*5)
  thin_data_list_agg[[i]] = do.call(rbind, thin_data_list[index]) %>% arrange(Date, Period)
}
sum(sapply(thin_data_list_agg, nrow) != 5465)
for (i in 1:nrow(edges)){
  thin_data_list_agg[[i]] = thin_data_list_agg[[i]][complete.cases(thin_data_list_agg[[i]]), ]
}
sum(sapply(thin_data_list_agg, nrow) != 5430)
nrow(edges) == length(thin_data_list_agg)
# 1825 - 1790 = 7*5 --> 7 NA for each period * 5 periods
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

identifiers %>% select(Date, Period) %>%  left_join(common_X, by = c("Date", "Period")) %>% write_csv("X_common_downtown.csv")
# temp1 = outcomes %>% select(starts_with("Y"))
# temp2 = outcomes %>% select(starts_with("Y"))
# for (i in 1:nrow(edges)){
#   origin = thin_data_list_agg[[i]]$origin[1]
#   destination = thin_data_list_agg[[i]]$destination[1]
#   stopifnot({
#     origin == edges[i, 1]
#     destination == edges[i, 2]
#   })
#   stopifnot(colnames(temp1)[i] == paste0("Y", "_origin", origin, "_destination", destination))
#   colnames(temp2)[i] = paste0("Y", "_origin", destination, "_destination", origin)
# }
# bind_cols(temp1, temp2) %>% write_csv("Y_downtown.csv")
outcomes %>% select(starts_with("Y")) %>% write_csv("Y_downtown.csv")
outcomes %>% select(starts_with("lag")) %>% write_csv("X_specific_downtown.csv")

temp = outcomes %>% select(starts_with("Y"))
for (i in 1:ncol(temp)){
  origin = edges[i, 1]
  destination = edges[i, 2]
  stopifnot(colnames(temp)[i] == paste0("Y", "_origin", origin, "_destination", destination))
}

X_specific = read_csv("X_specific_downtown.csv")
X_common = read_csv("X_common_downtown.csv")
bind_cols(X_common, X_specific) %>% write_csv("X_downtown.csv")
X = bind_cols(X_common, X_specific)
X %>% select(c("Period", "Temp", "WindSpeed",
               "Rain", "Visibility", "weekday", "month"),
             starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_thin_downtown.csv")

######## generate_A_and_b
edges = list()
edges[["1221"]] = make_small_matrix(1221, c(1222, 1220))
edges[["1222"]] = make_small_matrix(1222, c(1220))
edges[["1220"]] = make_small_matrix(1220 , c(1230, 1223, 1224, 1390))
edges[["1230"]] = make_small_matrix(1230 , c(1223, 1229, 1228, 1232, 1235))
edges[["1223"]] = make_small_matrix(1223 , c(1224, 1229))
edges[["1224"]] = make_small_matrix(1224 , c(1390, 1229))
edges[["1390"]] = make_small_matrix(1390 , c(1228, 1234, 1380))
edges[["1229"]] = make_small_matrix(1229 , c(1228))
edges[["1228"]] = make_small_matrix(1228 , c(1234, 1232, 1233))
edges[["1234"]] = make_small_matrix(1234 , c(1380, 1233))
edges[["1380"]] = make_small_matrix(1380 , c(1382))
edges[["1232"]] = make_small_matrix(1232 , c(1233, 1254))
edges[["1233"]] = make_small_matrix(1233 , c(1380, 1254, 1255, 1263))

edges[["1235"]] = make_small_matrix(1235 , c(1254, 1237))
edges[["1254"]] = make_small_matrix(1254 , c(1255, 1252, 1251))
edges[["1255"]] = make_small_matrix(1255 , c(1263, 1258))
edges[["1263"]] = make_small_matrix(1263 , c(1382, 1260, 1262))
edges[["1382"]] = make_small_matrix(1382 , c(1384))

edges[["1237"]] = make_small_matrix(1237 , c(1252, 1236, 1239))
edges[["1252"]] = make_small_matrix(1252 , c(1251, 1253))
edges[["1251"]] = make_small_matrix(1251 , c(1255, 1250, 1248))
edges[["1236"]] = make_small_matrix(1236 , c(1253, 1238))
edges[["1253"]] = make_small_matrix(1253 , c(1250, 1251))

edges[["1239"]] = make_small_matrix(1239 , c(1238, 1240))
edges[["1238"]] = make_small_matrix(1238 , c(1250, 1249, 1241))
edges[["1250"]] = make_small_matrix(1250 , c(1248, 1249))
edges[["1249"]] = make_small_matrix(1249 , c(1257, 1246))
edges[["1248"]] = make_small_matrix(1248 , c(1258, 1249))
edges[["1258"]] = make_small_matrix(1258 , c(1260, 1257))
edges[["1257"]] = make_small_matrix(1257 , c(1260, 1256))
edges[["1260"]] = make_small_matrix(1260 , c(1262, 1259))
edges[["1262"]] = make_small_matrix(1262 , c(1384, 1261))
edges[["1384"]] = make_small_matrix(1384 , c(1383))

edges[["1240"]] = make_small_matrix(1240 , c(1241, 1243))
edges[["1241"]] = make_small_matrix(1241 , c(1246, 1247, 1243))
edges[["1246"]] = make_small_matrix(1246 , c(1256, 1247))
edges[["1247"]] = make_small_matrix(1247 , c(1256, 1245))
edges[["1256"]] = make_small_matrix(1256 , c(1259))
edges[["1259"]] = make_small_matrix(1259 , c(1261))
edges[["1261"]] = make_small_matrix(1261 , c(1383))

edges[["1243"]] = make_small_matrix(1243 , c(1245, 1244, 1242))
edges[["1245"]] = make_small_matrix(1245 , c(1256, 1244))
edges[["1242"]] = make_small_matrix(1242 , c(1244))
edges = do.call(rbind, edges)
# temp = matrix(0, nrow(edges), 2)
# temp[, 2] = edges[, 1]
# temp[, 1] = edges[, 2]
# edges = rbind(edges, temp)


id = unique(c(edges))
A = matrix(0, nrow = length(id), ncol = nrow(edges))
# 1221 --> 1256, 1244, 1383
# 1222 --> 1256, 1244, 1383
source = 1221
end = 1256
A[which(id == source), edges[, 1] == source] = -1
A[which(id == end), edges[, 2] == end] = 1
for (k in 1:length(id)){
  if ((id[k] != source) & (id[k] != end)){
    A[k, edges[, 2] == id[k]] = 1
    A[k, edges[, 1] == id[k]] = -1
  }
}
b = rep(0, length(id))
b[which(id == source)] = -1
b[which(id == end)] = 1
write_csv(as.data.frame(A), "A_downtwon_1221to1256.csv")
write_csv(as.data.frame(b), "b_downtwon_1221to1256.csv")


path1 = c(0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path2 = c(0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path3 = c(0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path4 = c(0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path5 = c(0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path6 = c(0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path7 = c(0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path8 = c(0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path9 = c(0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path10 = c(0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path11 = c(0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
path12 = c(1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
paths = list(path1, path2, path3,
             path4, path5, path6,
             path7, path8, path9,
             path10, path11, path12)

path_vector = vector("list", length(paths))
for (i in 1:length(paths)){
  temp = edges[paths[[i]]==1, ]
  stopifnot(temp[1:(nrow(temp)-1), 2] == temp[2:nrow(temp), 1])
}
for (i in 1:length(paths)){
  path_vector[[i]] = c(edges[paths[[i]]==1, ][1, 1], edges[paths[[i]]==1, 2])
}


source = 1221
end = 1244
A[which(id == source), edges[, 1] == source] = -1
A[which(id == end), edges[, 2] == end] = 1
for (k in 1:length(id)){
  if ((id[k] != source) & (id[k] != end)){
    A[k, edges[, 2] == id[k]] = 1
    A[k, edges[, 1] == id[k]] = -1
  }
}
b = rep(0, length(id))
b[which(id == source)] = -1
b[which(id == end)] = 1
write_csv(as.data.frame(A), "A_downtwon_1221to1244.csv")
write_csv(as.data.frame(b), "b_downtwon_1221to1244.csv")

X = read_csv("X_downtown.csv")
X %>% filter((year(X$Date) == 2019)&(month(X$Date) >= 7)) %>%
  select(c("Period", "Temp", "WindSpeed",
           "Rain", "Visibility", "weekday", "month"),
         starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_halfyear.csv")
Y = read_csv("Y_downtown.csv")
Y %>% filter((year(X$Date) == 2019)&(month(X$Date) >= 7)) %>%
  write_csv("Y_halfyear.csv")

X = read_csv("X_downtown.csv")
X %>% filter(year(X$Date) == 2019) %>%
  select(c("Period", "Temp", "WindSpeed",
           "Rain", "Visibility", "weekday", "month"),
         starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_oneyear.csv")
Y = read_csv("Y_downtown.csv")
Y %>% filter(year(X$Date) == 2019) %>% write_csv("Y_oneyear.csv")


X %>% filter((year(X$Date) == 2019) | ( (year(X$Date) == 2018) & (month(X$Date) >= 7) )) %>%
  select(c("Period", "Temp", "WindSpeed",
           "Rain", "Visibility", "weekday", "month"),
         starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_onehalfyear.csv")
Y = read_csv("Y_downtown.csv")
Y %>% filter((year(X$Date) == 2019) | ( (year(X$Date) == 2018) & (month(X$Date) >= 7) )) %>%
  write_csv("Y_onehalfyear.csv")

X %>% filter((year(X$Date) == 2019) | (year(X$Date) == 2018)) %>%
  select(c("Period", "Temp", "WindSpeed",
           "Rain", "Visibility", "weekday", "month"),
         starts_with("lag1"), starts_with("lag7")) %>% write_csv("X_twoyear.csv")
Y = read_csv("Y_downtown.csv")
Y %>% filter((year(X$Date) == 2019) | (year(X$Date) == 2018)) %>% write_csv("Y_twoyear.csv")
