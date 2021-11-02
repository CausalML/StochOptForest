library(tidyverse)
library(lubridate)
library(riem)

make_small_matrix <- function(x, y){
  XX = rep(x, length(y))
  YY = y
  temp = cbind(XX, YY)
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

city="los_angeles"
sd_dates = c('2019-01-01', '2019-04-01', '2019-07-01', '2019-10-01',
             "2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01")
ed_dates = c('2019-03-31', '2019-06-30', '2019-09-30', '2019-12-31',
             '2018-03-31', '2018-06-30', '2018-09-30', '2018-12-31')
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

i = 1
j = 1
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


######## generate_A_and_b
id = unique(c(edges))
A = matrix(0, nrow = length(id), ncol = nrow(edges))
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
