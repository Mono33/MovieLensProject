#Author : Louis Mono
#Capstone1 :MovieLens project
#repository link : https://github.com/Mono33/MovieLensProject

##############################################################################################################

#This in a R code which generates predicted ratings and RMSE evaluations for the MovieLens project.


#In my repository, except this Capstone_Movielens_Rcode, you have the following files: 
#-Capstone_Movielens_data.R  which is the provided code for generating edx and validation sets
#-PDFReport_Movielens.pdf :  Movielens report in pdf format ( download it on your pc)
#-Report_Movielens.Rmd  : Movielens report in rmd file ;  you find the notebook here: https://mono33.github.io/MovieLensProject/ 

#but you also find:
#-EnvCapstone_Movielens.RData  which is the saving Rdata image of the Capstone_Movielens_data.R ( large file > 100MB then i didn't commit it )
#-EnvCapstone_MatrixFacto_trainRmse.RData which contains the first 30 iterations(trains recommender model) with rmse values for the optimized Matrix factorization.


##I.---------------------------------Section I: Introduction --------------------------------------------

#text on pdf and .rmd file

##II. ------------------------------- Section 2: Dataset and executive summary ----------------------------------------------------------

#1. Overview --------------------------------------------------------------------------------------

#load libraries
library(tidyverse)
library(caret)
library(data.table)
options(kableExtra.latex.load_packages = FALSE)
library(kableExtra)
library(lubridate)
library(Matrix.utils)
library(DT)
library(wordcloud) 
library(RColorBrewer) 
library(ggthemes) 
library(recommenderlab)
library(irlba)
library(SlopeOne)
library(recosystem)
library(h2o)



#download data

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#structure of training and validation datasets

glimpse(edx)
glimpse(validation)


# 2. Data exploration --------------------------------------------------------------------------------

#outcome, rating.

group <-  ifelse((edx$rating == 1 |edx$rating == 2 | edx$rating == 3 | 
                    edx$rating == 4 | edx$rating == 5) ,
                 "whole_star", 
                 "half_star") 

explore_ratings <- data.frame(edx$rating, group)


# histogram of ratings

ggplot(explore_ratings, aes(x= edx.rating, fill = group)) +
  geom_histogram( binwidth = 0.2) +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  scale_fill_manual(values = c("half_star"="purple", "whole_star"="brown")) +
  labs(x="rating", y="number of ratings", caption = "source data: edx set") +
  ggtitle("histogram : number of ratings for each rating")



# qualitative features:  genres, title

# genre

top_genr <- edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

datatable(top_genr, rownames = FALSE, filter="top", options = list(pageLength = 5, scrollX=T) ) %>%
  formatRound('count',digits=0, interval = 3, mark = ",")


layout(matrix(c(1,2), nrow =2) , heights = c(1,4))
par(mar=rep(0,4))
plot.new()
text(x=0.5,y=0.5, "top Genres by number of ratings")
wordcloud(words=top_genr$genres,freq=top_genr$count,min.freq=50,
          max.words = 20,random.order=FALSE,random.color=FALSE,
          rot.per=0.35,colors = brewer.pal(8,"Dark2"),scale=c(5,.2),
          family="plain",font=2,
          main = "Top genres by number of ratings")


#(highlight genres effects)
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 100000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "error bar plots by genres" , caption = "source data : edx set") +
  theme(
    panel.background = element_rect(fill = "lightblue",
                                    colour = "lightblue",
                                    size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                    colour = "white"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                    colour = "white")
  )

#title

top_title <- edx %>%
  group_by(title) %>%
  summarize(count=n()) %>%
  top_n(20,count) %>%
  arrange(desc(count))

# with the head function i output the top 5 

kable(head(edx %>%
             group_by(title,genres) %>%
             summarize(count=n()) %>%
             top_n(20,count) %>%
             arrange(desc(count)) ,
           5)) %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T) %>%
  column_spec(3,bold=T)

#bar chart of top_title

top_title %>% 
  ggplot(aes(x=reorder(title, count), y=count)) +
  geom_bar(stat='identity', fill="blue") + coord_flip(y=c(0, 40000)) +
  labs(x="", y="Number of ratings") +
  geom_text(aes(label= count), hjust=-0.1, size=3) +
  labs(title="Top 20 movies title based \n on number of ratings" , caption = "source data: edx set")



# quantitative features : userId, movieId , timestamp


# distinct users and movies
edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))



# histogram of number of ratings by movieId 

edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, color = "red") +
  scale_x_log10() + 
  ggtitle("Movies") +
  labs(subtitle  ="number of ratings by movieId", 
       x="movieId" , 
       y="number of ratings", 
       caption ="source data : edx set") +
  theme(panel.border = element_rect(colour="black", fill=NA)) 


# histogram of number of ratings by userId


edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, color = "gold") +
  scale_x_log10() + 
  ggtitle("Users") +
  labs(subtitle ="number of ratings by UserId", 
       x="userId" , 
       y="number of ratings") +
  theme(panel.border = element_rect(colour="black", fill=NA)) 


#Visual exploration of the number of ratings by movieId on one hand  and of the number of ratings by userId on the other hand shows the following relationships : some movies get rated more than others, and some users are more active than others at rating movies. 
#This should presumably explain the presence of movies effects and users effects.


#timestamp ( highlight time effects)
edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Timestamp, time unit : week")+
  labs(subtitle = "average ratings",
       caption = "source data : edx set")



##III.----------------------- Section 3: Data Preprocessing--------------------------------------------------------

#1.Data transformation 


#as.factor : As said in the Data exploration step,  usersId and movieId should be treat as factors for some analysis purposes

edx.copy <- edx
           
edx.copy$userId <- as.factor(edx$userId)
edx$movieId <- as.factor(edx$movieId)
           
           
#sparse matrix transformation
          
#first avoid, we need userId and MovieId  as numeric to perform it 
edx$userId <- as.numeric(edx$userId)
edx$movieId <- as.numeric(edx$movieId)
           
sparse_ratings <- sparseMatrix(i = edx$userId,
                              j = edx$movieId ,
                              x = edx$rating, 
                              dims = c(length(unique(edx$userId)),
                                       length(unique(edx$movieId))),  
                                       dimnames = list(paste("u", 1:length(unique(edx$userId)), sep = ""), 
                                                          paste("m", 1:length(unique(edx$movieId)), sep = "")))
           
           
#give a look on the first 10 users
sparse_ratings[1:10,1:10]
           
           
           
#Convert rating matrix into a recommenderlab sparse matrix
ratingMat <- new("realRatingMatrix", data = sparse_ratings)
ratingMat
           
           
           
#2.Similarity measures
           
#i calculate the user similarity using the cosine similarity
           
similarity_users <- similarity(ratingMat[1:50,], 
                               method = "cosine", 
                               which = "users")
           
image(as.matrix(similarity_users), main = "User similarity")
           
           
           
#Using the same approach, I compute similarity between  movies.
           
similarity_movies <- similarity(ratingMat[,1:50], 
                                method = "cosine", 
                                which = "items")
    
image(as.matrix(similarity_movies), main = "Movies similarity")
           
           
           
           
#3.Dimension reduction

set.seed(1)
Y <- irlba(sparse_ratings,tol=1e-4,verbose=TRUE,nv = 100, maxit = 1000)

# plot singular values
windows(title="")
plot(Y$d, pch=20, col = "blue", cex = 1.5, xlab='Singular Value', ylab='Magnitude', 
     main = "Singular Values for User-Movie Matrix")


# calculate sum of squares of all singular values
all_sing_sq <- sum(Y$d^2)

# variability described by first 6, 12, and 20 singular values
first_six <- sum(Y$d[1:6]^2)
print(first_six/all_sing_sq)

first_12 <- sum(Y$d[1:12]^2)
print(first_12/all_sing_sq)

first_20 <- sum(Y$d[1:20]^2)
print(first_20/all_sing_sq)

perc_vec <- NULL
for (i in 1:length(Y$d)) {
  perc_vec[i] <- sum(Y$d[1:i]^2) / all_sing_sq
}

plot(perc_vec, pch=20, col = "blue", cex = 1.5, xlab='Singular Value', ylab='% of Sum of Squares of Singular Values', main = "Choosing k for Dimensionality Reduction")
lines(x = c(0,100), y = c(.90, .90))



#To find the exact value of k, i calculate  the length of the vector that remains from our running sum of squares after excluding any items within that vector that exceed 0.90.

k = length(perc_vec[perc_vec <= .90])
K


#I get the decomposition of Y ; matrices U, D, and V accordingly:

U_k <- Y$u[, 1:k]
dim(U_k)

D_k <- Diagonal(x = Y$d[1:k])
dim(D_k)

V_k <- t(Y$v)[1:k, ]
dim(V_k)



#4.Relevant Data

#a.minimum number of movies per user
min_n_movies <- quantile(rowCounts(ratingMat), 0.9)
print(min_n_movies)
#b.minimum number of users per movie
min_n_users <- quantile(colCounts(ratingMat), 0.9)
print(min_n_users)
#c.Select the users and movies matching these criteria
ratings_movies <- ratingMat[rowCounts(ratingMat) > min_n_movies,
                            colCounts(ratingMat) > min_n_users]
ratings_movies



##IV----------------------- Section 4 : Methods and Analysis --------------------------------------------------------

#All references are shown in the rmd and pdf files . 
#i just report here the evaluation metric function (RMSE)

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


##v --------------------------------- Section 5: Results -----------------------------------------------------------

#I. Identifying optimal model


#1. Regression Models --------------------------------------------------------------------------------

#a.movie effect---

# i calculate the average of all ratings of the edx set
mu <- mean(edx$rating)

# i calculate b_i on the training set
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# predicted ratings
predicted_ratings_bi <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i


#b.movie + user effect ---

#i calculate b_u using the training set 
user_avgs <- edx %>%  
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#predicted ratings
predicted_ratings_bu <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred


#c.movie + user + time effect ---

#i create a copy of validation set , valid, and create the date feature which is the timestamp converted to a datetime object  and  rounded by week.

valid <- validation
valid <- valid %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) 

# i calculate time effects ( b_t) using the training set
temp_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))

# predicted ratings
predicted_ratings_bt <- valid %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(temp_avgs, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  .$pred

#d.  i calculate the RMSE for movies, users and time effects 

rmse_model1 <- RMSE(validation$rating,predicted_ratings_bi)  
rmse_model1
rmse_model2 <- RMSE(validation$rating,predicted_ratings_bu)
rmse_model2
rmse_model3 <- RMSE(valid$rating,predicted_ratings_bt)
rmse_model3
           

#Before to proceed with regularization, i just remove the object copy of validation, "valid"
rm(valid)

#e. regularization 

# remembering (5), $\lambda$ is a tuning parameter. We can use cross-validation to choose it

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu_reg <- mean(edx$rating)
  
  b_i_reg <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu_reg)/(n()+l))
  
  b_u_reg <- edx %>% 
    left_join(b_i_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_reg = sum(rating - b_i_reg - mu_reg)/(n()+l))
  
  predicted_ratings_b_i_u <- 
    validation %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg) %>%
    .$pred
  
  return(RMSE(validation$rating,predicted_ratings_b_i_u))
})


#For the full model, the optimal  λ is:

lambda <- lambdas[which.min(rmses)]
lambda

#rmse for Linear regression model with regularized movie and user effects
rmse_model4 <- min(rmses)
rmse_model4

df <- data.frame(lambdas,rmses)
#plot lambdas vs rmses : regularization
df %>% 
  ggplot(aes(x=lambdas, y=rmses))+
  geom_point()+
  geom_segment(x=0, xend=lambda,y=rmse_model4,yend=rmse_model4, color="red")+
  geom_segment(x=lambda, xend=lambda,y=0,yend=rmse_model4,color="red") +
  annotate(geom="label",x = lambda,y = 0.86485,color=2, label=paste("x=",round(lambda,2),"\ny=",round(rmse_model4,7)))  


#summarize all the rmse on validation set for Linear regression models

rmse_results <- data.frame(methods=c("movie effect","movie + user effects","movie + user + time effects", "Regularized Movie + User Effect Model"),rmse = c(rmse_model1, rmse_model2,rmse_model3, rmse_model4))

kable(rmse_results) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="#D7261E")





#2. Recommender engines--------------------------------------------------------------------------------


# a. POPULAR , UBCF and IBCF algorithms of the recommenderlab package

model_pop <- Recommender(ratings_movies, method = "POPULAR", 
                         param=list(normalize = "center"))

#prediction example on the first 10 users
pred_pop <- predict(model_pop, ratings_movies[1:10], type="ratings")
as(pred_pop, "matrix")[,1:10]

#Calculation of rmse for popular method 
set.seed(1)
e <- evaluationScheme(ratings_movies, method="split", train=0.7, given=-5)
#5 ratings of 30% of users are excluded for testing

model_pop <- Recommender(getData(e, "train"), "POPULAR")

prediction_pop <- predict(model_pop, getData(e, "known"), type="ratings")

rmse_popular <- calcPredictionAccuracy(prediction_pop, getData(e, "unknown"))[1]
rmse_popular


#Estimating rmse for UBCF using Cosine similarity and selected n as 50 based on cross-validation
set.seed(1)
model <- Recommender(getData(e, "train"), method = "UBCF", 
                     param=list(normalize = "center", method="Cosine", nn=50))

prediction <- predict(model, getData(e, "known"), type="ratings")

rmse_ubcf <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]
rmse_ubcf

#Estimating rmse for IBCF using Cosine similarity and selected n as 350 based on cross-validation
set.seed(1)

model <- Recommender(getData(e, "train"), method = "IBCF", 
                     param=list(normalize = "center", method="Cosine", k=350))

prediction <- predict(model, getData(e, "known"), type="ratings")

rmse_ibcf <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]
rmse_ibcf

#POPULAR, UBCF , IBCF methods don't fill with the scope of our study since the rmse evaluation is made on a test set after partitioning. We want to predict ratings for the 999999 rows and then evaluate with RMSE our close these predictions are with respect to the true ratings values in validation set.
#Moreover, the new data in the predict function should be of a class "ratingMatrix. Validation set doesn't have to be modified.We then moved to the SlopeOne and Matrix factorization methods 



#b. SlopeOne  -----------------------------------------------------------------------------------------

#Before to perform SlopeOne method, i clear unusued memory
invisible(gc())

# i create copy of training(edx) and validation sets where i retain only userId, movieId and rating
edx.copy <- edx %>%
  select(-c("genres","title","timestamp"))

valid.copy <- validation %>%
  select(-c("genres","title","timestamp"))

# i rename columns and convert them to characters  for edx.copy and valid.copy sets : item_id  is seen as movie_id

names(edx.copy) <- c("user_id", "item_id", "rating")

edx.copy <- data.table(edx.copy)

edx.copy[, user_id := as.character(user_id)]
edx.copy[, item_id := as.character(item_id)]


names(valid.copy) <- c("user_id", "item_id", "rating")

valid.copy <- data.table(valid.copy)

valid.copy[, user_id := as.character(user_id)]
valid.copy[, item_id := as.character(item_id)]


#setkey() sorts a data.table and marks it as sorted (with an attribute sorted). The sorted columns are the key. The key can be any columns in any order. The columns are sorted in ascending order always. The table is changed by reference and is therefore very memory efficient.
setkey(edx.copy, user_id, item_id)
setkey(valid.copy, user_id, item_id)


#split data to create a small training sample ( to face the RAM memory issue)
set.seed(1)
idx <- createDataPartition(y = edx.copy$rating, times = 1, p = 0.2, list = FALSE)
edx.copy_train <- edx.copy[idx,]

#normalization
ratings_train_norm <- normalize_ratings(edx.copy_train)

#Building Slope One model:
invisible(memory.limit(size = 56000))
model <- build_slopeone(ratings_train_norm$ratings)

#Making predictions for valdation set:

predictions <- predict_slopeone(model, 
                                valid.copy[ , c(1, 2), with = FALSE], 
                                ratings_train_norm$ratings)

unnormalized_predictions <- unnormalize_ratings(normalized = ratings_train_norm, 
                                                ratings = predictions)

#rmse - SlopeOne method
rmse_slopeone <- RMSE( valid.copy$rating,unnormalized_predictions$predicted_rating) 
rmse_slopeone

# i remove the created copies of sets
rm(edx.copy,valid_copy,edx.copy_train)


#c. Matrix Factorization with parallel stochastic gradient descent-------------------------------------


#Before to perform MF method, i clear unusued memory
invisible(gc())


#i create a copy of training(edx) and validation sets where i retain only userId, movieId and rating features. i rename the three columns.

edx.copy <-  edx %>%
  select(-c("genres","title","timestamp"))

names(edx.copy) <- c("user", "item", "rating")


valid.copy <-  validation %>%
  select(-c("genres","title","timestamp"))

names(valid.copy) <- c("user", "item", "rating")


#as matrix
edx.copy <- as.matrix(edx.copy)
valid.copy <- as.matrix(valid.copy)


#write edx.copy and valid.copy tables on disk 
write.table(edx.copy , file = "trainset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
write.table(valid.copy, file = "validset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)


#  data_file(): Specifies a data set from a file in the hard disk. 

set.seed(123) # This is a randomized algorithm
train_set <- data_file(system.file( "dat" ,"trainset.txt" , package = "recosystem"))
valid_set <- data_file(system.file( "dat" ,"validset.txt" , package = "recosystem"))


#Next step is to build Recommender object
r = Reco()


#'Matrix Factorization : tuning parameters  with  default nfolds = 5'
opts = r$tune(train_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                     costp_l1 = 0, costq_l1 = 0,
                                     nthread = 1, niter = 10))

opts


#' Matrix Factorization : trains the recommender model '
r$train(train_set, opts = c(opts$min, nthread = 1, niter = 20))



#Making prediction on validation set and calculating RMSE:
pred_file = tempfile()
r$predict(valid_set, out_file(pred_file))

#'Matrix Factorization : show first 10 predicted values'
print(scan(pred_file, n = 10))


#valid_set
scores_real <- read.table("validset.txt", header = FALSE, sep = " ")$V3
scores_pred <- scan(pred_file)

# rmse - Matrix factorization method
rmse_mf <- RMSE(scores_real,scores_pred)
rmse_mf

# i remove the created copies of sets
rm(edx.copy, valid.copy)



#summarize all the rmse on validation set for recommender algorithms

rmse_results <- data.frame(methods=c("SlopeOne","Matrix factorization with GD"),rmse = c(rmse_slopeone, rmse_mf))

kable(rmse_results) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold =T ,color = "white" , background ="#D7261E")


#3. Ensemble Methods ---------------------------------------------------------------------------------


# i create a copy of edx set where i retain all the features
edx.copy <- edx

# i create new columns n.movies_byUser (number of movies each user rated) and n.users_bymovie (Number of users that rated each movie)  as defined in the Data analysis and method section
edx.copy <- edx.copy %>%
  group_by(userId) %>%
  mutate(n.movies_byUser = n())

edx.copy <- edx.copy %>%
  group_by(movieId) %>%
  mutate(n.users_bymovie = n())

# factor vectors of users and movies
edx.copy$userId <- as.factor(edx.copy$userId)
edx.copy$movieId <- as.factor(edx.copy$movieId)

#i do the same for the validation set
valida.copy <- validation  

valida.copy <- valida.copy %>%
  group_by(userId) %>%
  mutate(n.movies_byUser = n())

valida.copy <- valida.copy %>%
  group_by(movieId) %>%
  mutate(n.users_bymovie = n())

valida.copy$userId <- as.factor(valida.copy$userId)
valida.copy$movieId <- as.factor(valida.copy$movieId)


#Attempts to start and/or connect to and H2O instance

h2o.init(
  nthreads=-1,             ## -1: use all available threads
  max_mem_size = "10G")    ## specify the memory size for the H2O cloud


#partitioning 
splits <- h2o.splitFrame(as.h2o(edx.copy), 
                         ratios = 0.7, 
                         seed = 1) 
train <- splits[[1]]
test <- splits[[2]]


#first gbm model :  ntrees = 50, max depth = 5, learn rate = 0.1 , nfolds = 3 
gbdt_1 <- h2o.gbm( x = c("movieId","userId","n.movies_byUser","n.users_bymovie") ,
                   y = "rating" , 
                   training_frame = train , 
                   nfolds = 3)

summary(gbdt_1)

#second gbm model :  ntrees = 100, max depth = 5, learn rate = 0.1 , nfolds = 5 
gbdt_2 <- h2o.gbm( x = c("movieId","userId") ,
                   y = "rating" , 
                   training_frame = train , 
                   ntrees = 100,
                   nfolds = 5) 

summary(gbdt_2)

#third gbm model :  ntrees = 50, max depth = 5, learn rate = 0.1 , nfolds = 3 
gbdt_3 <- h2o.gbm( x = c("movieId","userId") ,
                   y = "rating" , 
                   training_frame = train , 
                   nfolds = 3,
                   seed=1,
                   keep_cross_validation_predictions = TRUE,
                   fold_assignment = "Random") 

summary(gbdt_3)

#third gbm model :  ntrees = 50, max depth = 5, learn rate = 0.1 , nfolds = 3 
gbdt_3 <- h2o.gbm( x = c("movieId","userId") ,
                   y = "rating" , 
                   training_frame = train , 
                   nfolds = 3) 

summary(gbdt_3)


#Since the model gbdt_3  has the lower RMSE on training set,   

# i evaluate performance on test set
h2o.performance(gbdt_3, test)

#i predict ratings on validation set and evaluate RMSE
pred.ratings.gbdt_3 <- h2o.predict(gbdt_3,as.h2o(valida.copy))

rmse_gbdt <- RMSE(pred.ratings.gbdt_3, as.h2o(valida.copy$rating))
rmse_gbdt


# first rf model : 
rf1 <- h2o.randomForest(        
  training_frame = train,       
  x= c("movieId" ,"userId" ,"timestamp", "n.movies_byUser","n.users_bymovie"),                  
  y= "rating",                         
  ntrees = 50,                 
  max_depth = 20
)

summary(rf1)

# second rf model : ntrees = 50, max.deptu = 20 ,  nfolds = 5
rf2 <- h2o.randomForest(        
  training_frame = train,       
  x= c("movieId" ,"userId", "n.movies_byUser","n.users_bymovie"),                      
  y= "rating",                         
  ntrees = 50,                 
  max_depth = 20,
  nfolds = 5
)

summary(rf2)


#third rf model : ntrees = 50, max.depth = 20 , nfolds = 3
rf_3 <- h2o.randomForest(        
  training_frame = train,       
  x= c("movieId" ,"userId"),                      
  y= "rating", 
  nfolds=3,
  seed=1,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Random"
)


#Since the model rf_3  has the lower RMSE on training set,   

# i evaluate performance on test set
h2o.performance(rf_3, test)

#i predict ratings on validation set and evaluate RMSE
pred.ratings.rf_3 <- h2o.predict(rf_3,as.h2o(valida.copy))

rmse_rf <- RMSE(pred.ratings.rf_3, as.h2o(valida.copy$rating))
rmse_rf


#stacked Ensemble : i take the best two previous model (gbdt_3 and rf_3)

ensemble <- h2o.stackedEnsemble(x = c("movieId" ,"userId"),
                                y = "rating",
                                training_frame = train,
                                model_id = "my_ensemble_auto",
                                base_models = list(gbdt_3@model_id, rf_3@model_id))

#i predict ratings on validation set and evaluate RMSE
pred.ratings.ensemble <- h2o.predict(ensemble,as.h2o(valida.copy))

rmse_ensemble <- RMSE(pred.ratings.ensemble, as.h2o(valida.copy$rating))
rmse_ensemble


# i remove the created copies of sets
rm(edx.copy,valida.copy)

#summarize all the rmse on validation set for ensemble methods

rmse_results <- data.frame(methods=c("gradient Boosting","random forest","stacked ensemble"),rmse = c(rmse_gbdt, rmse_rf, rmse_ensemble))

kable(rmse_results) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold = T ,color = "white" , background ="#D7261E")


#close the cluster h2o
h2o.shutdown()



#II. Increasing model performance

#see preamble text on pdf report 

#i clear unusued memory
invisible(gc())

#i create a copy of training(edx) and validation sets where i retain only userId, movieId and rating features. i rename the three columns.

edx.copy <-  edx %>%
  select(-c("genres","title","timestamp"))

names(edx.copy) <- c("user", "item", "rating")


valid.copy <-  validation %>%
  select(-c("genres","title","timestamp"))

names(valid.copy) <- c("user", "item", "rating")


#as matrix
edx.copy <- as.matrix(edx.copy)
valid.copy <- as.matrix(valid.copy)


#write edx.copy and valid.copy tables on disk 
write.table(edx.copy , file = "trainset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
write.table(valid.copy, file = "validset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)


#data_file(): Specifies a data set from a file in the hard disk. 

set.seed(123) # This is a randomized algorithm
train_set <- data_file(system.file( "dat" ,"trainset.txt" , package = "recosystem"))
valid_set <- data_file(system.file( "dat" ,"validset.txt" , package = "recosystem"))


#Next step is to build Recommender object
r = Reco()


#Optimizing/tuning the recommender model
opts <- r$tune(train_set , opts = list(dim = c(1:20), lrate = c(0.05),
                                       nthread = 4 , costp_l1=0, 
                                       costq_l1 = 0,
                                       niter = 100, nfold = 10,
                                       verbose = FALSE))

#trains the recommender model
r$train(train_set, opts = c(opts$min , nthread = 4, niter = 100,verbose=FALSE))


#Making prediction on validation set and calculating RMSE:
pred_file = tempfile()

r$predict(valid_set, out_file(pred_file))  

#show first 10 predicted values
print(scan(pred_file, n = 10))


#valid_set
scores_real <- read.table("validset.txt", header = FALSE, sep = " ")$V3
scores_pred <- scan(pred_file)

# remove edx.copy and valid.copy objects
rm(edx.copy, valid.copy)

#rmse - increasing performance MF method
rmse_mf_opt <- sqrt(mean((scores_real-scores_pred) ^ 2))
rmse_mf_opt  


# load the Rdata  which contains first 30 iterations of the trains recommender model. Then, plot the smooth curve
# of the number of latent factors vs. cross-validation RMSE

load("EnvCapstone_MatrixFacto_trainRmse.RData")

iter.line <- 15
tr_rmse.line <- mat.facto_rmse$tr_rmse[which(mat.facto_rmse$iter==15)]


windows(title="")
mat.facto_rmse %>% 
  ggplot(aes(x=iter, y = tr_rmse))+
  geom_point(size= 5 , shape = 19 ) + 
  geom_smooth(aes(x= iter, y = tr_rmse)) +
  geom_segment(x=0,xend=iter.line ,y=tr_rmse.line,yend=tr_rmse.line, color="red", lty=2)+
  geom_segment(x=iter.line, xend=iter.line, y=0, yend=tr_rmse.line, color="red", lty=2) +
  annotate(geom="label",x = iter.line,y = 0.8350,color=2,     
           label=paste("x=",round(iter.line,0),"\ny=",round(tr_rmse.line ,4)))+
  labs(title="RMSE for different number of latent factors" ,
       caption = "based on the output of r$train(train_set, opts = c(opts$min, nthread = 4, niter = 100), \n show just first 30 iterations)"
  ) +
  ylab("RMSE") +
  xlab("Latent factors")



#VI--------------------- Section 6 : Conclusion and suggestions----------------------------------------

#text on pdf and rmd file



####################################THANKS TO READ THIS RSCRIPT AND EDIT IF NECESSARY#################################################################################

