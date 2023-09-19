# load libraries
# data wrangling
library(dplyr)
library(tidyr)
library(lubridate)
# visualization
library(ggplot2)
# dealing with text
library(corpus)
library(textclean)
library(tm)
library(udpipe)
library(SnowballC)
library(stringr)
library(lattice)
library(Matrix)
# topic model
library(tidytext)
library(topicmodels)
library(textmineR)

# setup environment
setwd("C:\Users\Cecilia\OneDrive\Desktop\Ceci\UniTS\Introduction_to_Machine_Learning\Progetto\carteggio.svevo.csv")

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
stats <- subset(x, upos %in% c("NOUN", "VERB")) 
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
dtm_matrix_colsum_perc_sub <- dtm_matrix_colsum_perc[dtm_matrix_colsum_perc < 5]
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

## Most occurring lemmas of type noun, verb, adj after the reduction of terms
x_treshold <- subset(x, lemma %in% dtf_threshold$term)
stats <- subset(x_treshold, upos %in% c("NOUN")) 
stats <- txt_freq(stats$lemma)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 40), col = "cadetblue", 
         main = "Most occurring nouns, verbs, adj (lemma)", xlab = "Freq")

####################################################################################

#evaluation

# https://www.tidytextmining.com/topicmodeling.html

### apply lda
##tm <- LDA(dtm_threshold, k = 10, control = list(seed = 1234))
##str(tm)
##topics <- tidy(tm, matrix = "beta")
##head(topics)
##
##top_terms <- topics %>%
## group_by(topic) %>%
## slice_max(beta, n = 10) %>% 
## ungroup() %>%
## arrange(topic, -beta)
##
##top_terms %>%
##  mutate(term = reorder_within(term, beta, topic)) %>%
##  ggplot(aes(beta, term, fill = factor(topic))) +
##  geom_col(show.legend = FALSE) +
##  facet_wrap(~ topic, scales = "free") +
##  scale_y_reordered()


##################################################################################

#evaluation

# https://slcladal.github.io/topicmodels.html

### apply lda
##dim(dtm_threshold)
### due to vocabulary pruning, we have empty rows in our DTM
### LDA does not like this. So we remove those docs from the
### DTM and the metadata
##sel_idx <- slam::row_sums(dtm_threshold) > 0
##dtm_threshold <- dtm_threshold[sel_idx, ]
##dim(dtm_threshold)
##
### number of topics
##K <- 20
### set random number generator seed
##set.seed(9161)
### compute the LDA model, inference via 1000 iterations of Gibbs sampling
##topicModel <- LDA(dtm_threshold, K, method="Gibbs", control=list(iter = 500, verbose = 25))
##str(topicModel)
### have a look a some of the results (posterior distributions)
##tmResult <- posterior(topicModel)
### format of the resulting object
##attributes(tmResult)
### topics are probability distribtions over the entire vocabulary
##beta <- tmResult$terms   # get beta from results
##dim(beta)                # K distributions over nTerms(DTM) terms
##rowSums(beta)            # rows in beta sum to 1
### for every document we have a probaility distribution of its contained topics
##theta <- tmResult$topics 
##dim(theta)               # nDocs(DTM) distributions over K topics
##rowSums(theta)[1:10]     # rows in theta sum to 1
###Let's take a look at the 10 most likely terms within the term probabilities
###beta of the inferred topics
##exampleTermData <- terms(topicModel, 10)
##top5termsPerTopic <- terms(topicModel, 5)
##topicNames <- apply(top5termsPerTopic, 2, paste, collapse=" ")
##
### visualize topics as word cloud
##topicToViz <- 11 # change for your own topic of interest
##topicToViz <- grep('mexico', topicNames)[1] # Or select a topic by a term contained in its name
### select to 40 most probable terms from the topic by sorting the term-topic-probability vector in decreasing order
##top40terms <- sort(tmResult$terms[topicToViz,], decreasing=TRUE)[1:40]
##words <- names(top40terms)
### extract the probabilites of each of the 40 terms
##probabilities <- sort(tmResult$terms[topicToViz,], decreasing=TRUE)[1:40]
### visualize the terms as wordcloud
##mycolors <- brewer.pal(8, "Dark2")
####wordcloud(words, probabilities, random.order = FALSE, color = mycolors) ## capire perchè non va
##
##
##
### get topic proportions
##N <- dim(theta)[1]
##topicProportionExamples <- theta
##colnames(topicProportionExamples) <- topicNames
##vizDataFrame <- melt(cbind(data.frame(topicProportionExamples), document = factor(1:N)), variable.name = "topic", id.vars = "document")  
##ggplot(data = vizDataFrame, aes(topic, value, fill = document), ylab = "proportion") + 
##  geom_bar(stat="identity") +
##  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +  
##  coord_flip() +
##  facet_wrap(~ document, ncol = N)

#####################################################################################################################

## Evaluation

# https://medium.com/swlh/topic-modeling-in-r-with-tidytext-and-textminer-package-latent-dirichlet-allocation-764f4483be73

# apply to LDA function. set the k = 10, means we want to build 10 topic 
lda_ceci <- LDA(dtm_threshold,k = 10,control = list(seed = 1234))
# apply auto tidy using tidy and use beta as per-topic-per-word probabilities
topic_ceci <- tidy(lda_ceci,matrix = "beta")
# choose 15 words with highest beta from each topic
top_terms <- topic_ceci %>%
  group_by(topic) %>%
  top_n(15,beta) %>% 
  ungroup() %>%
  arrange(topic,-beta)
# plot the topic and words for easy interpretation
plot_topic <- top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered()
plot_topic


# other method usefull to choose best topics:
set.seed(1234)
mod_lda_ceci <- FitLdaModel(dtm = dtm_threshold,
                         k = 10, # number of topic
                         iterations = 500,
                         burnin = 180,
                         alpha = 0.1,beta = 0.05,
                         optimize_alpha = T,
                         calc_likelihood = T,
                         calc_coherence = T,
                         calc_r2 = T)
#R-squared is interpretable as the proportion of
#variability in the data explained by the model, as with linear regression.
mod_lda_ceci$r2
plot(mod_lda_ceci$log_likelihood,type = "l")

#get 15 top terms with the highest phi. phi representing a distribution of words
#over topics. Words with high phi have the most frequency in a topic.
mod_lda_ceci$top_terms <- GetTopTerms(phi = mod_lda_ceci$phi,M = 15)
data.frame(mod_lda_ceci$top_terms)

#coherence value for each topic. Topic Coherence measures score a single topic by
#measuring the degree of semantic similarity between high scoring words in the
#topic. In simple words, coherence tell us how associated words are in a topic
mod_lda_ceci$coherence

#Prevalence tells us the most frequent topics in the corpus.
#Prevalence is the probability of topics distribution in the whole documents.
mod_lda_ceci$prevalence <- colSums(mod_lda_ceci$theta)/sum(mod_lda_ceci$theta)*100
mod_lda_ceci$prevalence

#summary
mod_lda_ceci$summary <- data.frame(topic = rownames(mod_lda_ceci$phi),
                                coherence = round(mod_lda_ceci$coherence,3),
                                prevalence = round(mod_lda_ceci$prevalence,3),
                                top_terms = apply(mod_lda_ceci$top_terms,2,function(x){paste(x,collapse = ", ")}))

modsum_ceci <- mod_lda_ceci$summary %>%
  `rownames<-`(NULL)

#know that the quality of the model can be described with coherence and prevalence
#value. let's build a plot to identify which topic has the best quality
modsum_ceci %>% pivot_longer(cols = c(coherence,prevalence)) %>%
  ggplot(aes(x = factor(topic,levels = unique(topic)), y = value, group = 1)) +
  geom_point() + geom_line() +
  facet_wrap(~name,scales = "free_y",nrow = 2) +
  theme_minimal() +
  labs(title = "Best topics by coherence and prevalence score",
       x = "Topics", y = "Value")

#We can see if topics can be grouped together using Dendogram.
#A Dendrogram uses Hellinger distance (distance between 2 probability vectors)
#to decide if the topics are closely related.
mod_lda_ceci$linguistic <- CalcHellingerDist(mod_lda_ceci$phi)
mod_lda_ceci$hclust <- hclust(as.dist(mod_lda_ceci$linguistic),"ward.D")
mod_lda_ceci$hclust$labels <- paste(mod_lda_ceci$hclust$labels, mod_lda_ceci$labels[,1])
plot(mod_lda_ceci$hclust)

#We will choose 5 different topic with the highest quality (coherence).
#Each topic will have 15 words with highest value of phi
#(distribution of words over topics).
modsum_ceci %>% 
  arrange(desc(coherence)) %>%
  slice(1:5)















