# load libraries
library("corpus")
library("tm")
library(udpipe)
library(lattice)
library(Matrix)
library(topicmodels)

# setup environment
setwd("G:\\Il mio Drive\\ML Project 2021\\text_mining_svevo")

# load data
# Encoded in ANSI
data <- read.csv2("carteggio.svevo3_ANSI.csv", header=TRUE)
#head(data)
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

##########################################################################################################

### Data Pre-Processing
# convert text to compatible format
mytext <- Corpus(VectorSource(c_data$text))
str(mytext)

# remove punctuation UNICODE charset
new_text <- tm_map(mytext, removePunctuation, preserve_intra_word_contractions = TRUE,
                                             preserve_intra_word_dashes = TRUE,
                                             ucp = TRUE)
#head(new_text$content)

#Transform to lower case
new_text <- tm_map(new_text, content_transformer(tolower))
#head(new_text$content)

#Strip digits
new_text <- tm_map(new_text, removeNumbers)
#head(new_text$content)

# Remove all special characters
f <- content_transformer(function(x) gsub("[[:punct:]]", " ", x))
new_text <- tm_map(new_text, f)
#head(new_text$content)

#Strip Whitespaces
new_text <- tm_map(new_text, stripWhitespace)
#head(new_text$content)

#Remove stop words
# new_text <- tm_map(new_text, removeWords, c(stopwords_it, "ch"))
new_text <- tm_map(new_text, removeWords, stopwords_it)
new_text <- tm_map(new_text, removeWords, stopwords_en)
new_text <- tm_map(new_text, removeWords, stopwords_de)
new_text <- tm_map(new_text, removeWords, stopwords_fr)
new_text <- tm_map(new_text, removeWords, c("ch"))
#head(new_text$content)
#new_text$content[1]
#c_data$text[1]

#Remove date related words
#my_dates <- c("gennaio", "febbraio", "marzo", "aprile",
              #  "maggio", "giugno", "luglio", "agosto", 
              #  "settembre", "ottobre", "novembre", "dicembre",
              #  "lunedì","martedì","mercoledì","giovedì",
              # "venerdì","sabato","domenica")

#new_text <- tm_map(new_text, removeWords, c(my_months,my_dates))
#head(new_text$content)


#Strip Whitespaces
new_text <- tm_map(new_text, stripWhitespace)
#head(new_text$content)

# add modified text as new dataframe column
c_data$edited <- new_text$content
head(c_data$edited)

# POS tagging
#ud_model <- udpipe_download_model(language = "italian")
#ud_model <- udpipe_load_model(ud_model$file_model)
#x <- udpipe_annotate(ud_model, x = new_text$content, doc_id = c_data$n)

x <- udpipe(c_data$edited, "italian", doc_id = c_data$n)
head(x)


#x_new_text <- udpipe(head(new_text$content, 100), "italian", doc_id = head(c_data$n, 100))

# UPOS (Universal Parts of Speech) frequency 
stats <- txt_freq(x$upos)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = stats, col = "cadetblue", 
         main = "UPOS (Universal Parts of Speech)\n frequency of occurrence", 
         xlab = "Freq")


## Most occurring lemmas of type noun, verb, adj
stats <- subset(x, upos %in% c("NOUN", "VERB","ADJ")) 
stats <- txt_freq(stats$lemma)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 40), col = "cadetblue", 
         main = "Most occurring nouns, verbs, adj (lemma)", xlab = "Freq")

# how many times a term occurred in each letter
sub_x <- subset(x, upos %in% c("NOUN", "VERB","ADJ")) 
dtf <- document_term_frequencies(sub_x, document = "doc_id", term = "lemma")
head(dtf)


# Create a document/term/matrix for building a topic model
dtm <- document_term_matrix(x = dtf)
head(dtm)

# compute for 
dtm_matrix <- as.matrix(dtm)
dtm_matrix <- dtm_matrix/dtm_matrix
dtm_matrix_colsum <- sort(colSums(dtm_matrix, na.rm=TRUE), decreasing = TRUE)
head(dtm_matrix_colsum)
dtm_matrix_colsum_perc <- (dtm_matrix_colsum/826)*100
head(dtm_matrix_colsum_perc, 40)

# exclude terms which recur too much: in more than 10% of the letters
dtm_matrix_colsum_perc_sub <- dtm_matrix_colsum_perc[dtm_matrix_colsum_perc < 10]
head(dtm_matrix_colsum_perc_sub)
tail(dtm_matrix_colsum_perc_sub)

# exclude terms which recur in not enough letters: in less than 1% of the letters
dtm_matrix_colsum_perc_threshold <- dtm_matrix_colsum_perc_sub[dtm_matrix_colsum_perc_sub > 1]
tail(dtm_matrix_colsum_perc_threshold)

# exclude from dtf all terms non present in dtm_matrix_colsum_perc_threshold
names(dtm_matrix_colsum_perc_threshold)
head(dtf)
dtf_threshold <- dtf[dtf$term %in% names(dtm_matrix_colsum_perc_threshold) ]
head(dtf_threshold)
#length(unique(dtf_threshold$term))
#length(names(dtm_matrix_colsum_perc_threshold))

# filtered dtm 
dtm_threshold <- document_term_matrix(x = dtf_threshold)
head(dtm_threshold)

# apply lda
tm <- LDA(dtm_threshold, k = 50)
str(tm)


