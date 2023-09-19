# load libraries
library("corpus")
library("tm")

# setup environment
setwd("G:\\Il mio Drive\\ML Project 2021\\text_mining_svevo")

# load data
# Encoded in ANSI
data <- read.csv2("carteggio.svevo3_ANSI.csv", header=TRUE)
head(data)
str(data)

# format data
# covert to corpus
c_data <- corpus_frame(data)
str(c_data)
attach(c_data)

# convert languages to factor
c_data$languages <- as.factor(c_data$languages)
c_data$mainLanguage <- as.factor(c_data$mainLanguage)
c_data$corpus <- as.factor(c_data$corpus)
c_data$sender <- as.factor(c_data$sender)
c_data$recipient <- as.factor(c_data$recipient)
str(c_data)

# languages levels
languages_factor <-levels(c_data$languages)
# stat of languages per letter
summary(c_data$languages)
# histogram on languages frequencies
barplot(sort(summary(c_data$languages)))
# most frequent languages (combinations): ITA, ITA-UND, ITA-FRE
# ITA reference language
 
summary(c_data$mainLanguage)
# histogram on languages frequencies
barplot(sort(summary(c_data$mainLanguage)))
# ITA most frequent MAIN language

# DISCARD LETTERS WITH MAIN LANGUAGE DIFFERENT FROM ITALIAN
c_data <- subset(c_data, c_data$mainLanguage=="ITA")
c_data$mainLanguage <- factor(c_data$mainLanguage)
summary(c_data$mainLanguage)
# left: 826 letters with ITA as main language

# looking for missing data or anomalies
levels(c_data$sender)
levels(c_data$recipient)
sort(c_data$year)

### Data Pre-Processing
# convert text to compatible format
mytext <- Corpus(VectorSource(c_data$text))
str(mytext)

# remove punctuation
new_text <- tm_map(mytext, removePunctuation)
#Transform to lower case
docs <- tm_map(docs, content_transformer(tolower))



# add modified text as new dataframe column
c_data$new_text <- new_text$content
head(c_data)




