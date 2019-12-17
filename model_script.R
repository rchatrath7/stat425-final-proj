library(MASS)
library(tidyverse)
library(caret)
library(doParallel)
library(mltools)
library(zoo)
#library(Rfast)
library(ggplot2)
library(sigmoid)

train_raw <- read_csv("data/train.csv.zip")
weather <- read_csv("data/weather.csv")

test_raw <- read_csv("data/test.csv")
key <- read.csv("data/key.csv")

join_table = function(input, key, weather) {
  return(
    input %>%  
      left_join(key, by = "store_nbr") %>% 
      inner_join(weather, by = c("station_nbr" = "station_nbr", "date" = "date")) %>% 
      select(-station_nbr)
  )
}

train <- join_table(train_raw, key, weather)
test <- join_table(test_raw, key, weather)

train_typed <- mutate(train, id = as.factor(paste(store_nbr, item_nbr, date, sep = "_"))) %>% 
  mutate(units = as.numeric(units)) %>%  
  filter(units > 0) %>% 
  mutate(
    date = as.numeric(date), 
    tmin = as.numeric(tmin), 
    tmax = as.numeric(tmax), 
    tavg = as.numeric(tavg), 
    depart = as.numeric(depart), 
    dewpoint = as.numeric(dewpoint), 
    wetbulb = as.numeric(wetbulb), 
    heat = as.numeric(heat), 
    cool = as.numeric(cool), 
    sunrise = as.factor(sunrise), 
    sunset = as.factor(sunset), 
    codesum = as.factor(codesum), 
    snowfall = as.numeric(snowfall), 
    preciptotal = as.numeric(preciptotal), 
    stnpressure = as.numeric(stnpressure), 
    sealevel = as.numeric(sealevel), 
    resultspeed = as.numeric(resultspeed), 
    resultdir = as.numeric(resultdir), 
    avgspeed = as.numeric(avgspeed), 
    store_nbr = as.factor(store_nbr), 
    item_nbr = as.factor(item_nbr)
  )

test_typed <- mutate(test, id = as.factor(paste(store_nbr, item_nbr, date, sep = "_"))) %>% 
  mutate(
    date = as.numeric(date), 
    tmin = as.numeric(tmin), 
    tmax = as.numeric(tmax), 
    tavg = as.numeric(tavg), 
    depart = as.numeric(depart), 
    dewpoint = as.numeric(dewpoint), 
    wetbulb = as.numeric(wetbulb), 
    heat = as.numeric(heat), 
    cool = as.numeric(cool), 
    sunrise = as.factor(sunrise), 
    sunset = as.factor(sunset), 
    codesum = as.factor(codesum), 
    snowfall = as.numeric(snowfall), 
    preciptotal = as.numeric(preciptotal), 
    stnpressure = as.numeric(stnpressure), 
    sealevel = as.numeric(sealevel), 
    resultspeed = as.numeric(resultspeed), 
    resultdir = as.numeric(resultdir), 
    avgspeed = as.numeric(avgspeed), 
    store_nbr = as.factor(store_nbr), 
    item_nbr = as.factor(item_nbr)
  )

save(train_typed, file = "data/train.Rda")
save(test_typed, file = "data/test.Rda")

load("data/train.Rda")
load("data/test.Rda")

trn = train_typed %>% 
  select(-codesum, -id, -sunset, -sunrise, -heat, -cool)

cl <- makePSOCKcluster(8)
registerDoParallel(cl)

base_linear_mod = train(
  units ~ ., 
  data = trn, 
  method = "lm", 
  na.action = na.pass, 
)

stopCluster(cl)

p = predict(base_linear_mod, test_typed)

plot(fitted(base_linear_mod), resid(base_linear_mod), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Data from Model 1")
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(base_linear_mod))
qqline(resid(base_linear_mod))

sub = tibble(
  id = test_typed$id
)

pred = tibble(
  units = p, 
  idx = as.numeric(names(p))
)

sub_join = sub %>% mutate(idx = as.numeric(rownames(sub))) %>% left_join(pred, by = 'idx')
sub_join = sub_join %>% 
  mutate(units = replace_na(units, 0)) %>% 
  select(-idx)

sub_join %>% 
  write.csv('data/submission.csv', row.names = FALSE)

boxcox(units ~ ., data = trn, lambda = seq(-.5, .5, .01))

bx = BoxCoxTrans(trn$units) 

cd = cooks.distance(base_linear_mod$finalModel)
length(cd[cd > 4 * mean(cd)])

hist(trn$units)
hist(log1p(trn$units))
hist(predict(bx, trn$units))

log_linear_mod = train(
  log1p(units) ~ ., 
  data = trn, 
  method = "lm", 
  na.action = na.pass
)

p = predict(log_linear_mod, test_typed)

sub = tibble(
  id = test_typed$id
)

pred = tibble(
  units = p, 
  idx = as.numeric(names(p))
)

sub_join = sub %>% mutate(idx = as.numeric(rownames(sub))) %>% left_join(pred, by = 'idx')
sub_join = sub_join %>% 
  mutate(units = replace_na(units, 0))
sub_join = sub_join %>% dplyr::select(-idx)

sub_join %>% 
  write.csv('data/submission.csv', row.names = FALSE)

plot(fitted(log_linear_mod), resid(log_linear_mod), col = "grey", pch = 20,
     xlab = "Fitted", ylab = ")Residuals", main = "Data from Model 1")
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(log_linear_mod))
qqline(resid(log_linear_mod))

cdl = cooks.distance(log_linear_mod$finalModel)
length(cdl[cdl > 4 / length(cdl)])

trn_out = trn %>% filter(!row_number() %in% which(cdl > 4 / length(cdl)))

log_linear_mod_outrm = train(
  log1p(units) ~ ., 
  data = trn_out, 
  method = "lm", 
  na.action = na.pass
)

plot(fitted(log_linear_mod_outrm), resid(log_linear_mod_outrm), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Log Linear, Out RM")
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(log_linear_mod_outrm))
qqline(resid(log_linear_mod_outrm))

p = predict(log_linear_mod_outrm, test_typed)


sub = tibble(
  id = test_typed$id
)

pred = tibble(
  units = p, 
  idx = as.numeric(names(p))
)

sub_join = sub %>% mutate(idx = as.numeric(rownames(sub))) %>% left_join(pred, by = 'idx')
sub_join = sub_join %>% 
  mutate(units = replace_na(units, 0))
sub_join = sub_join %>% dplyr::select(-idx)

sub_join %>% 
  write.csv('data/submission.csv', row.names = FALSE)

td = dummyVars(units ~ ., data = trn %>% select)
trn_hot = as_tibble(predict(td, trn))
trn_hot$units = trn$units
tst_hot = as_tibble(predict(td, test_typed))
