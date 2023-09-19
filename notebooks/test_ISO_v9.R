# load libraries
# data wrangling
library(dplyr)
library(tidyr)
library(lubridate)
# visualization
library(ggplot2)
library(networkD3)
library(RColorBrewer)
# dealing with text
library(corpus)
library(textclean)
library(tm)
library(udpipe)
library(SnowballC)
library(stringr)
library(lattice)
# usefull libraries
library(rlist)
library(Matrix)
# topic model
library(tidytext)
library(topicmodels)
library(textmineR)
library(Hmisc)

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
#new_text <- tm_map(new_text, content_transformer(tolower))
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

# add modified text as new dataframe column
c_data$edited <- new_text$content

# Try POS tagging first
x <- udpipe(c_data$edited, "italian-partut", doc_id = c_data$n)
head(x)

## Most occurring lemmas of type noun, verb, adj
stats <- subset(x, upos %in% c("NOUN", "VERB","PROPN")) 
stats <- txt_freq(stats$lemma)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 40), col = "cadetblue", 
         main = "Most occurring nouns, verbs, adj (lemma)", xlab = "Freq")

# how many times a term occurred in each letter
sub_x <- subset(x, upos %in% c("NOUN", "VERB","PROPN")) 

sub_x <- subset(sub_x, !(token %in% stopwords_it)) 
sub_x <- subset(sub_x, !(token %in% capitalize(stopwords_it))) 
sub_x <- subset(sub_x, !(token %in% c('schmitz', 'signore', 'signora', 'mano', 'mani', 'ettore', 'lettera', 'parola', 'fare', 'cosa', 'caro', 'cara'))) 
sub_x <- subset(sub_x, !(token %in% capitalize(c('schmitz', 'signore', 'signora', 'mano', 'mani','ettore', 'lettera', 'parola', 'fare', 'cosa', 'caro', 'cara')))) 

#Remove date related words
my_dates <- c("gennaio", "febbraio", "marzo", "aprile",
               "maggio", "giugno", "luglio", "agosto", 
                "settembre", "ottobre", "novembre", "dicembre",
                "lunedì","lunedi","martedì","martedi","mercoledì","mercoledi", "giovedì", "giovedi",
                "venerdì","venerdi","sabato","domenica")
sub_x <- subset(sub_x, !(token %in% my_dates))
sub_x <- subset(sub_x, !(token %in% capitalize(my_dates)))

# Remove short words
#sort(nchar(sub_x$token))
sub_x <- subset(sub_x, nchar(token) > 2)


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



################################################################################
#                      FUNZIONI PER PARAMETER TUNING
################################################################################

# Alpha parameter range
alpha <- c(0.005,0.01,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3)
# Beta parameter range
beta <- seq(0.1, 1, 0.1)
# Topics parameter rang
k_int <- seq(3, 10, 1)

cohe_tab <- data.frame(matrix(ncol = 4, nrow = 0))
model_list <- list()
start_time <- Sys.time()

for(k in k_int){
  for(b in beta){
    for(a in alpha){
      lda_model_tuned <- FitLdaModel(dtm = dtm_threshold,
                                     k = k, # number of topic
                                     iterations = 300, #GIBBS iterations
                                     alpha = a,
                                     beta = b, 
                                     calc_likelihood = T,
                                     calc_coherence = T,
                                     calc_r2 = T)
      if (mean(lda_model_tuned$coherence) > 0.15) {
        lda_model_tuned$Alpha <- a
        lda_model_tuned$Beta <- b
        lda_model_tuned$K <- k
        model_list <- list.append(model_list,lda_model_tuned)
        new_row <- c(k,b,a,mean(lda_model_tuned$coherence))
        cohe_tab <- rbind(cohe_tab, new_row)
      }
    }
  }
}

end_time <- Sys.time()
tempo <- end_time-start_time
colnames(cohe_tab) <- c("K","Beta","Alpha","Coherence")

save.image(file='new_pos_tagging_upper_env.RData')

# view models
GetTopTerms(phi = model_list[[14]]$phi,M = 20)
GetTopTerms(phi = model_list[[58]]$phi,M = 20)
GetTopTerms(phi = model_list[[12]]$phi,M = 30)
GetTopTerms(phi = model_list[[8]]$phi,M = 20)


model_li

model_max$top_terms <- GetTopTerms(phi = model_list[[3]]$phi,M = 20)
data.frame(model_max$top_terms)

model_list_lda <- list()
choe_tab_lda <- data.frame()
start_time_lda <- Sys.time()
for (k in k_int) {
  model_lda <- LDA(dtm_threshold, k, method = "Gibbs", control = list(seed = 1234,iter=300))
  #topic_ldas <- tidy(model_lda, matrix = "beta")
  model_list_lda <- list.append(model_list_lda,model_lda)
  print(terms(model_lda,10))
}
end_time_lda <- Sys.time()

terms(model_list_lda[[2]],10)



#################################################################################################
#                                         Final Analysis                                        #
#################################################################################################

final_data_topic <- final_data[c(1,5,6,8,14,15,16,17,18)]

# un argomento viene considerato in una lettera solo se è trattato con percentuale superiore al 20% (messo a 0)
final_data_topic$t_1[final_data_topic$t_1 < 0.2] <- 0
final_data_topic$t_1[final_data_topic$t_1 >= 0.2] <- 1
final_data_topic$t_2[final_data_topic$t_2 < 0.2] <- 0
final_data_topic$t_2[final_data_topic$t_2 >= 0.2] <- 1
final_data_topic$t_3[final_data_topic$t_3 < 0.2] <- 0
final_data_topic$t_3[final_data_topic$t_3 >= 0.2] <- 1
final_data_topic$t_4[final_data_topic$t_4 < 0.2] <- 0
final_data_topic$t_4[final_data_topic$t_4 >= 0.2] <- 1
final_data_topic$t_5[final_data_topic$t_5 < 0.2] <- 0
final_data_topic$t_5[final_data_topic$t_5 >= 0.2] <- 1

final_data_topic$t_1[is.na(final_data_topic$t_1)] <- 0
final_data_topic$t_2[is.na(final_data_topic$t_2)] <- 0
final_data_topic$t_3[is.na(final_data_topic$t_3)] <- 0
final_data_topic$t_4[is.na(final_data_topic$t_4)] <- 0
final_data_topic$t_5[is.na(final_data_topic$t_5)] <- 0

head(final_data_topic)

years <- sort(unique(final_data_topic$year))
topic_year <- data.frame(matrix(ncol = 3, nrow = 0))
str(topic_year)
for (i in years) {
  new_row_1 <- c(i,"t1",sum(final_data_topic$t_1[final_data_topic$year == i]))
  new_row_2 <- c(i,"t2",sum(final_data_topic$t_2[final_data_topic$year == i]))
  new_row_3 <- c(i,"t3",sum(final_data_topic$t_3[final_data_topic$year == i]))
  new_row_4 <- c(i,"t4",sum(final_data_topic$t_4[final_data_topic$year == i]))
  new_row_5 <- c(i,"t5",sum(final_data_topic$t_5[final_data_topic$year == i]))
  topic_year <- rbind(topic_year, new_row_1, new_row_2, new_row_3, new_row_4, new_row_5)}

colnames(topic_year) <- c("year","topic","total")
head(topic_year)

# create nodes dataframe
nodes <- data.frame(node = c(0:(length(years)+4)), 
                    name = c(years, "t1", "t2", "t3", "t4", "t5"))
head(nodes)
#create links dataframe
topic_year_new <- merge(topic_year, nodes, by.x = "year", by.y = "name")
head(topic_year_new)
topic_year_new <- merge(topic_year_new, nodes, by.x = "topic", by.y = "name")
head(topic_year_new)
links <- topic_year_new[ , c("node.x", "node.y", "total","topic","year")]
colnames(links) <- c("target", "source", "value","group","year")
head(links)

nodes$group <- as.factor(c("my_unique_group"))
nodes

# draw sankey network
display.brewer.all(colorblindFriendly = TRUE)
display.brewer.pal(n = 5, name = 'Dark2')

palette3 <- brewer.pal(n = 6, name = "Dark2")
palette3

my_color <- JS('d3.scaleOrdinal() .domain(["t1","t2","t3","t4","t5","my_unique_group"]) .range(["#1B9E77" "#D95F02" "#7570B3" "#E7298A" "#66A61E" "#E6AB02"])')
#my_color <- 'd3.scaleOrdinal() .domain(["type_a", "type_b", "my_unique_group"]) .range(palette3)'


networkD3::sankeyNetwork(Links = links, Nodes = nodes, 
                         Source = 'source', 
                         Target = 'target', 
                         Value = 'value', 
                         NodeID = 'name',
                         #units = 'votes',
                         #colourScale = my_color,
                         LinkGroup="group", NodeGroup="group",
                         fontSize = 12, nodeWidth = 12, 
                         fontFamily = "sans-serif", iterations = 0
                         )

# da fare la stessa cosa per le persone

# creare matrice topic person
# create poi le matrici nodes e links per darle poi in pasto con quella rete
# vedi se riesci a scegliere i colori per i collegamenti





















