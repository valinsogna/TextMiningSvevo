################################################################################
#                           ENV SETUP
################################################################################
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

setwd("G:\\Il mio Drive\\ML Project 2021\\text_mining_svevo")


################################################################################
#                           DATA LOAD AND EXPLORATION
################################################################################
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



################################################################################
#                      DATA PRE-PROCESSING
################################################################################

# convert text to compatible format
mytext <- Corpus(VectorSource(c_data$text))
str(mytext)

# remove punctuation UNICODE charset
new_text <- tm_map(mytext, removePunctuation, preserve_intra_word_contractions = TRUE,
                                             preserve_intra_word_dashes = TRUE,
                                             ucp = TRUE)

#Strip digits
new_text <- tm_map(new_text, removeNumbers)

# Remove all special characters
f <- content_transformer(function(x) gsub("[[:punct:]]", " ", x))
new_text <- tm_map(new_text, f)

#Strip Whitespaces
new_text <- tm_map(new_text, stripWhitespace)

# add modified text as new dataframe column
c_data$edited <- new_text$content

# POS tagging partut
# x <- udpipe(c_data$edited, "italian-partut", doc_id = c_data$n)
# POS tagging italian
x <- udpipe(c_data$edited, "italian", doc_id = c_data$n)
# POS tagging italian
# x <- udpipe(c_data$edited, "italian-isdt", doc_id = c_data$n)

head(x)

# Restrict lemmas to "NOUN", "VERB","PROPN"
sub_x <- subset(x, upos %in% c("NOUN", "VERB","PROPN")) 

# Remove stop words both starting with lower and upper case
# Default stop words
sub_x <- subset(sub_x, !(token %in% stopwords_it)) 
sub_x <- subset(sub_x, !(token %in% capitalize(stopwords_it))) 
# Custom stop words
sub_x <- subset(sub_x, !(token %in% c('schmitz', 'signore', 'signora', 'mano', 'mani', 'ettore', 'lettera', 'parola', 'fare', 'cosa', 'caro', 'cara'))) 
sub_x <- subset(sub_x, !(token %in% capitalize(c('schmitz', 'signore', 'signora', 'mano', 'mani','ettore', 'lettera', 'parola', 'fare', 'cosa', 'caro', 'cara')))) 
my_dates <- c("gennaio", "febbraio", "marzo", "aprile",
               "maggio", "giugno", "luglio", "agosto", 
                "settembre", "ottobre", "novembre", "dicembre",
                "lunedì","lunedi","martedì","martedi","mercoledì","mercoledi", "giovedì", "giovedi",
                "venerdì","venerdi","sabato","domenica")
sub_x <- subset(sub_x, !(token %in% my_dates))
sub_x <- subset(sub_x, !(token %in% capitalize(my_dates)))

# Remove short words
sub_x <- subset(sub_x, nchar(token) > 2)

# Plot most occurring lemmas of type noun, verb, propn
stats <- txt_freq(sub_x$lemma)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 40), col = "cadetblue", 
         main = "Most occurring nouns, verbs, adj (lemma)", xlab = "Freq")

# Create document/term/frequency
dtf <- document_term_frequencies(sub_x, document = "doc_id", term = "lemma")
head(dtf)

# Create a document/term/matrix for building a topic model
dtm <- document_term_matrix(x = dtf)
head(dtm)

# Compute terms occurring frequency 
dtm_matrix <- as.matrix(dtm)
dtm_matrix <- dtm_matrix/dtm_matrix
dtm_matrix_colsum <- sort(colSums(dtm_matrix, na.rm=TRUE), decreasing = TRUE)
head(dtm_matrix_colsum)
dtm_matrix_colsum_perc <- (dtm_matrix_colsum/nrow(c_data))*100
head(dtm_matrix_colsum_perc, 40)

# exclude terms which recur too much (more than 5% of the letters)
dtm_matrix_colsum_perc_sub <- dtm_matrix_colsum_perc[dtm_matrix_colsum_perc < 5]
head(dtm_matrix_colsum_perc_sub)
tail(dtm_matrix_colsum_perc_sub)

# exclude terms which recur in not enough letters: in less than 1% of the letters
dtm_matrix_colsum_perc_threshold <- dtm_matrix_colsum_perc_sub[dtm_matrix_colsum_perc_sub > 1]
tail(dtm_matrix_colsum_perc_threshold)

# update dtf removing all terms non present in dtm_matrix_colsum_perc_threshold
names(dtm_matrix_colsum_perc_threshold)
head(dtf)
dtf_threshold <- dtf[dtf$term %in% names(dtm_matrix_colsum_perc_threshold) ]
head(dtf_threshold)

# update dtm 
dtm_threshold <- document_term_matrix(x = dtf_threshold)
head(dtm_threshold)

## Most occurring lemmas of type noun, verb, propn after filtering
stats <- subset(sub_x, lemma %in% dtf_threshold$term)
stats <- txt_freq(stats$lemma)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 40), col = "cadetblue", 
         main = "Most occurring nouns, verbs, propn (lemma)", xlab = "Freq")



################################################################################
#                      PARAMETER TUNING
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

#save.image(file='pre_pos_tagging_upper_italian_isdt_5_perc_env.RData')

# print models with highest coherence 
for (i in head(order(cohe_tab$Coherence, decreasing=TRUE),5)){
  print(c("Model",i))
  print(c("Coherence = ",cohe_tab$Coherence[i]))
  print(GetTopTerms(phi = model_list[[i]]$phi,M = 30))
}

################################################################################
#                     BEST COHERENCE MODEL Analysis                    
################################################################################

# Select max coherence model
model_max <- model_list[[ 33 ]]
model_max$top_terms <- GetTopTerms(phi = model_max$phi,M = 30)
model_max$top_terms

#coherence value for each topic. Topic Coherence measures score a single topic by
#measuring the degree of semantic similarity between high scoring words in the
#topic. In simple words, coherence tell us how associated words are in a topic
model_max$coherence

#Prevalence tells us the most frequent topics in the corpus.
#Prevalence is the probability of topics distribution in the whole documents.
model_max$prevalence <- colSums(model_max$theta)/sum(model_max$theta)*100
model_max$prevalence


#summary
model_max$summary <- data.frame(topic = rownames(model_max$phi),
                                coherence = round(model_max$coherence,3),
                                prevalence = round(model_max$prevalence,3),
                                top_terms = apply(model_max$top_terms,2,function(x){paste(x,collapse = ", ")}))

plot_cohe_prev <- model_max$summary %>%
  `rownames<-`(NULL)

#know that the quality of the model can be described with coherence and prevalence
#value. let's build a plot to identify which topic has the best quality
plot_cohe_prev %>% pivot_longer(cols = c(coherence,prevalence)) %>%
  ggplot(aes(x = factor(topic,levels = unique(topic)), y = value, group = 1)) +
  geom_point() + geom_line() +
  facet_wrap(~name,scales = "free_y",nrow = 2) +
  theme_minimal() +
  labs(title = "Best topics by coherence and prevalence score",
       x = "Topics", y = "Value")


################################################################################
#                             RESULTS PREPARATION                      
################################################################################

# Threshold per matrice Theta (prob of topics in each document)
# frequency topic per letter
model_max$assignments <- model_max$theta
model_max$assignments[ model_max$assignments < 0.05 ] <- 0
model_max$assignments <- model_max$assignments / rowSums(model_max$assignments)
model_max$assignments[ is.na(model_max$assignments) ] <- 0

# Estrai la colonna dell'ID della lettera da theta
colonna <- dimnames(model_max$assignments)[[1]]
n <- as.integer(colonna)

# Aggiungi colonna ID lettera a theta e trasforma tutto in dataframe
my_assign <- cbind(model_max$assignments, n)
new_assign <- as.data.frame.matrix(my_assign)
head(new_assign)

#Aggiungi colonne con topics e valori di theta al file c_data (dataFrame con sola lingua IT senza preprocessing e senza POS)
final_data <- merge(c_data, new_assign, by = c("n"), all = T) 
#ATT: mette NA nelle colonne quando trova le lettere (cioÃ¨ righe) mancanti in new_assign

#ID lettere mancanti in new_assign
setdiff(c_data$n, new_assign$n)

################################################################################
#                             RESULTS ANALYSIS                      
################################################################################

final_data_topic <- final_data

# un argomento viene considerato in una lettera solo se è trattato con percentuale superiore al 20% (messo a 0)
final_data_topic$t_1[final_data_topic$t_1 < 0.2] <- 0
final_data_topic$t_1[final_data_topic$t_1 >= 0.2] <- 1
final_data_topic$t_2[final_data_topic$t_2 < 0.2] <- 0
final_data_topic$t_2[final_data_topic$t_2 >= 0.2] <- 1
final_data_topic$t_3[final_data_topic$t_3 < 0.2] <- 0
final_data_topic$t_3[final_data_topic$t_3 >= 0.2] <- 1
final_data_topic$t_4[final_data_topic$t_4 < 0.2] <- 0
final_data_topic$t_4[final_data_topic$t_4 >= 0.2] <- 1
#final_data_topic$t_5[final_data_topic$t_5 < 0.2] <- 0
#final_data_topic$t_5[final_data_topic$t_5 >= 0.2] <- 1

final_data_topic$t_1[is.na(final_data_topic$t_1)] <- 0
final_data_topic$t_2[is.na(final_data_topic$t_2)] <- 0
final_data_topic$t_3[is.na(final_data_topic$t_3)] <- 0
final_data_topic$t_4[is.na(final_data_topic$t_4)] <- 0
#final_data_topic$t_5[is.na(final_data_topic$t_5)] <- 0

head(final_data_topic)

years <- sort(unique(final_data_topic$year))
topic_year <- data.frame(matrix(ncol = 3, nrow = 0))
str(topic_year)
for (i in years) {
  new_row_1 <- c(i,"Letteratura",sum(final_data_topic$t_1[final_data_topic$year == i]))
  new_row_2 <- c(i,"Viaggiare",sum(final_data_topic$t_2[final_data_topic$year == i]))
  new_row_3 <- c(i,"Tempo libero",sum(final_data_topic$t_3[final_data_topic$year == i]))
  new_row_4 <- c(i,"Relazioni umane",sum(final_data_topic$t_4[final_data_topic$year == i]))
  #new_row_5 <- c(i,"t5",sum(final_data_topic$t_5[final_data_topic$year == i]))
  topic_year <- rbind(topic_year, new_row_1, new_row_2, new_row_3, new_row_4)
}

colnames(topic_year) <- c("year","topic","total")
head(topic_year)

# create nodes dataframe
nodes <- data.frame(node = c(0:(length(years)+3)), 
                    name = c(years, "Letteratura", "Viaggiare", "Tempo libero", "Relazioni umane"))
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
display.brewer.pal(n = 4, name = 'Dark2')

palette3 <- brewer.pal(n = 6, name = "Dark2")
palette3

my_color <- JS('d3.scaleOrdinal() .domain(["Letteratura", "Viaggiare", "Tempo libero", "Relazioni umane","my_unique_group"]) .range(["#1B9E77" "#D95F02" "#7570B3" "#E7298A" "#66A61E"])')
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


####### ASSOCIATION PERSON-TOPIC
# group by person (corpus)
pt_df <- data.frame(corpus = final_data_topic$corpus,
                    t_1 = final_data_topic$t_1,
                    t_2 = final_data_topic$t_2,
                    t_3 = final_data_topic$t_3,
                    t_4 = final_data_topic$t_4)
pt_group_df <- aggregate(cbind(t_1,t_2,t_3,t_4) ~ corpus, pt_df, sum)

# expand group by person dataframe
person_topic <- data.frame(matrix(ncol = 3, nrow = 0))
pt_group_df$corpus <- as.character(pt_group_df$corpus)
for (i in 1:nrow(pt_group_df)){
  r1 <- c(pt_group_df$corpus[i],"Letteratura",pt_group_df$t_1[i])
  r2 <- c(pt_group_df$corpus[i],"Viaggiare",pt_group_df$t_2[i])
  r3 <- c(pt_group_df$corpus[i],"Tempo libero",pt_group_df$t_3[i])
  r4 <- c(pt_group_df$corpus[i],"Relazioni umane",pt_group_df$t_4[i])
  person_topic <- rbind(person_topic,r1,r2,r3,r4)
}

colnames(person_topic) <- c("person","topic","total")

# reformat corpus names removng forst word which is always svevo/schmitz
person <- as.character(person_topic$person)
for (i in 1:length(person)){
  div <- strsplit(person[i], " ")
  div <- div[[1]][-1]
  person[i] <- str_trim(paste(div, collapse = " "))
}
person_topic$person <- person
head(person_topic$person)
# create nodes dataframe
nodes <- data.frame(
                    name = c(as.character(person_topic$person), 
                             "Letteratura", "Viaggiare", "Tempo libero",
                             "Relazioni umane")%>% unique()
                    )
# Add a 'group' column to each node. Here I decide to put all of them in the same group to make them grey
nodes$group <- as.factor(c("my_unique_group"))

links <- person_topic

links$IDsource <- match(links$person, nodes$name)-1 
links$IDtarget <- match(links$topic, nodes$name)-1
# Add a 'group' column to each connection:
links$group <- as.factor(links$topic)
str(links)

# Give a color for each group:
#my_color <- 'd3.scaleOrdinal() .domain(["Letteratura", "Viaggiare", "Tempo libero",
#                             "Relazioni umane", "my_unique_group"]) .range(["aquamarine",
#                             "burlywood1", "darkolivegreen2", "lightpink2", "black"])'

# colors <- tribble(
#   ~topic, ~color,
#   "Letteratura",    "aquamarine",
#   "Viaggiare",    "burlywood1",
#   "Tempo libero", "darkolivegreen2",
#   "Relazioni umane", "lightpink2"
# )

links$group <- sub(' .*', '',
                                nodes[links$IDsource + 1, 'name'])

# p <- networkD3::sankeyNetwork(Links = links, Nodes = nodes,
#                    Source = "IDsource", Target = "IDtarget",
#                    Value = "total", NodeID = "name", 
#                    sinksRight=FALSE, 
#                    fontSize = 12, nodeWidth = 12, 
#                    fontFamily = "sans-serif", iterations = 0,
#                    colourScale=my_color, LinkGroup="group", NodeGroup="group")

p <- networkD3::sankeyNetwork(Links = links, Nodes = nodes,
                              Source = "IDsource", Target = "IDtarget",
                              Value = "total", NodeID = "name", 
                              sinksRight=FALSE, 
                              fontSize = 12, nodeWidth = 12, 
                              fontFamily = "sans-serif", iterations = 0,
                              LinkGroup="group", NodeGroup=NULL)
p
