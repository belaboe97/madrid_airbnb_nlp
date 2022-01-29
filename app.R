library(dplyr)
library(superml)
library(utf8)
library(NLP)
library(tm)
library(qdapDictionaries)
library(tidytext)
library(stringi)
library(devtools)
library(cld2)
library(cld3)
library(spacyr)
library(stringr)
library(RWeka)
library(ggplot2)
library(wordcloud)


#Import data
data = read.csv("C:/Users/Bela Boente/Desktop/Programming/NLP/archive/reviews_detailed.csv",encoding = "UTF-8")

#head(data,1)
#summary(data)

data = sample_n(data,500)

#print(data)

#Encoding(data$comments)

utf8_encoded = iconv(data$comments, 'UTF-8', "ASCII")

pre_data = na.omit(utf8_encoded)

pre_data_en = pre_data[detect_language(pre_data) == "en" ]

#print(pre_data_en)

pre_data_en = na.omit(pre_data_en)

length(pre_data_en)


doc.vec = VectorSource(pre_data_en)
doc.corpus = VCorpus(doc.vec)

#https://rpubs.com/tahirhussa/207317
#Check this functions
#https://stackoverflow.com/questions/24771165/r-project-no-applicable-method-for-meta-applied-to-an-object-of-class-charact
#Convert to lower case


#Read more about V-Corpus
#https://rpubs.com/anlope10/588192

doc.corpus<- tm_map(doc.corpus, tolower)

#Remove all punctuatins

doc.corpus<- tm_map(doc.corpus, removePunctuation)
#Remove all numbers

doc.corpus<- tm_map(doc.corpus, removeNumbers)
#Remove whitespace

doc.corpus <- tm_map(doc.corpus, stripWhitespace)
#force everything back to plaintext document

#Adjust this part!

uniGramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
biGramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
triGramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
quadGramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 4, max = 4))

uniGramMatrix <- TermDocumentMatrix(doc.corpus, control = list(tokenize = uniGramTokenizer))
biGramMatrix <- TermDocumentMatrix(doc.corpus, control = list(tokenize = biGramTokenizer))
triGramMatrix <- TermDocumentMatrix(doc.corpus, control = list(tokenize = triGramTokenizer))
quadGramMatrix <- TermDocumentMatrix(doc.corpus, control = list(tokenize = quadGramTokenizer))

freqTerms <- findFreqTerms(biGramMatrix, lowfreq = 10)
termFrequency <- rowSums(as.matrix(biGramMatrix[freqTerms,]))
termFrequency <- data.frame(bigram=names(termFrequency), frequency=termFrequency)

UnifreqTerms <- findFreqTerms(uniGramMatrix, lowfreq = 500)
UnitermFrequency <- rowSums(as.matrix(uniGramMatrix[UnifreqTerms,]))
UnitermFrequency <- data.frame(unigram=names(UnitermFrequency), frequency=UnitermFrequency)

g1 <- ggplot(UnitermFrequency, aes(x=reorder(unigram, frequency), y=frequency)) +
  geom_bar(stat = "identity") +  coord_flip() +
  theme(legend.title=element_blank()) +
  xlab("Unigram") + ylab("Frequency") +
  labs(title = "Top Unigrams by Frequency")
print(g1)


####Second Plot 


g2 <- ggplot(termFrequency, aes(x=reorder(bigram, frequency), y=frequency )) +
  geom_bar(stat = "identity", colour = "red") +  coord_flip() +
  theme(legend.title=element_blank()) +
  xlab("Bigram") + ylab("Frequency") +
  labs(title = "Top Bigrams by Frequency ")
print(g2)


###Third Plot 
freqTerms <- findFreqTerms(triGramMatrix, lowfreq = 15)
termFrequency <- rowSums(as.matrix(triGramMatrix[freqTerms,]))
termFrequency <- data.frame(trigram=names(termFrequency), frequency=termFrequency)

g3 <- ggplot(termFrequency, aes(x=reorder(trigram, frequency), y=frequency)) +
  geom_bar(stat = "identity", colour = "blue") +  coord_flip() +
  theme(legend.title=element_blank()) +
  xlab("Trigram") + ylab("Frequency") +
  labs(title = "Top Trigrams by Frequency")
print(g3)


###Fourth Plot 

freqTerms <- findFreqTerms(quadGramMatrix, lowfreq = 15)
termFrequency <- rowSums(as.matrix(quadGramMatrix[freqTerms,]))
termFrequency <- data.frame(quadgram=names(termFrequency), frequency=termFrequency)


g4 <- ggplot(termFrequency, aes(x=reorder(quadgram, frequency), y=frequency)) +
  geom_bar(stat = "identity", colour = "blue") +  coord_flip() +
  theme(legend.title=element_blank()) +
  xlab("Quadgram") + ylab("Frequency") +
  labs(title = "Top Quadgrams by Frequency")
print(g4)

#wordcloud(doc.corpus, max.words = 500, random.order = FALSE,rot.per=0.35, use.r.layout=FALSE,colors=brewer.pal(6, "Dark2"))

#https://www.pluralsight.com/guides/machine-learning-text-data-using-r



library(reticulate)
library(tensorflow)
#https://stackoverflow.com/questions/62532838/how-to-install-keras-bert-packagesnotfounderror-the-following-packages-are-no
#reticulate::use_condaenv("bert_env", required=TRUE)

reticulate::use_python('C:/tools/Anaconda3/python.exe',required=T)

reticulate::py_config()

reticulate::py_module_available("keras_bert")

tensorflow::tf_config()

reticulate::py_install('genism', pip = TRUE)

k_bert = import('keras_bert')




token_dict = k_bert$load_vocabulary(vocab_path)
tokenizer = k_bert$Tokenizer(token_dict)

tokenizer.encode(pre_data_en)

library(LiblineaR)
library(tidymodels)

reticulate::py_install('transformers', pip = TRUE)
transformer = reticulate::import('transformers')


tf = reticulate::import('tensorflow')


#https://huggingface.co/bert-base-uncased import 
tokenizer = transformer$BertTokenizer$from_pretrained('bert-base-uncased')
model = transformer$TFBertForMaskedLM$from_pretrained('bert-base-uncased')


text = pre_data_en[20]

encoding =tokenizer$encode(text)

print(encoding)


print(tokenizer$convert_ids_to_tokens(encoding))




mutate(pre_data_en, tolower())

df = data.frame(pre_data_en)

pre_data_1 = df %>% mutate(excerpt=gsub("[^[:alnum:][:space:].]", "",substr(df,1,nchar(df)-1)))

pre_data_1 =  pre_dat_1  %>% mutate(excerpt= tolower(excerpt))

pre_data_1
stop


































#cleaned_text <- filter(str_detect(pre_data_en, "^[^>]+[A-Za-z\\d]") | pre_data_en !="") 

text = paste(unlist(pre_data_en[1:length(pre_data_en)]), collapse="\n")[1]

text

#Basic checks from HandsOn 1/3
text[!utf8_valid(text)]
#character(0) -> valid utf-8
text_NFC = utf8_normalize(text)
sum(text_NFC != text)
# Result 0 -> valid 

stringText <- paste(text, collapse = "\n") 
paragraphs <- unlist(strsplit(stringText, "\\n\\n\\n"))

parEmpty <- which(paragraphs == "") 
parEmpty



paragraphswoNL <- gsub("[\n]{1,}", " ", paragraphs)

length(text)
print(text)

cleaned_text = text %>%
  filter(str_detect(text, "^[^>]+[A-Za-z\\d]") | text !="") 






sentences_part1 = spacy_tokenize(text, what="sentence") #Returns a list
v_sentences_part1 <- unlist(sentences_part1)

#stringi::stri_trans_general(utf8_encoded, "latin-ascii")

data %>% 
  unnest(comments) %>% unique %>% select(comments) -> comments_txt

Encoding(comments_txt[1])

#Taking a sample 

#https://rpubs.com/tahirhussa/207317
cleaned_txt = iconv(comments_txt, 'UTF-8', 'ASCII', "byte")
print(cleaned_txt[1])

cleaned_txt = tolower(cleaned_txt)
cleaned_txt = removeWords(cleaned_txt,stopwords('en'))

text_df = tibble(id_review = cleaned_txt)

text_df = text_df %>%  unnest_tokens(word, id_review)

cleaned_txt %>% unnest_tokens(words, y, to_lower = T) %>% 
  mutate(text = words %in% GradyAugmented) 

text_df


text_df = text_df[text_df %in% GradyAugmented]

print(text_df)

print(cleaned_txt)

#Checking if lines are utf8 valid 
comments_txt[!utf8_valid(comments_txt)]



library("textcat")
textcat(c(
  "This is an English sentence.",
  "Das ist ein deutscher Satz.",
  "Esta es una frase en espa~nol."))
