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
library(sbo)



#Import data
data = read.csv("C:/Users/Bela Boente/Desktop/Programming/NLP/archive/reviews_detailed.csv",encoding = "UTF-8")

#head(data,1)
#summary(data)

data = sample_n(data,5000)

#print(data)

#Encoding(data$comments)

utf8_encoded = iconv(data$comments, 'UTF-8', "ASCII")

pre_data = na.omit(utf8_encoded)

pre_data_en = pre_data[detect_language(pre_data) == "en" ]

print(pre_data_en)

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

#doc.corpus<- tm_map(doc.corpus, removePunctuation)
#Remove all numbers

doc.corpus<- tm_map(doc.corpus, removeNumbers)
#Remove whitespace

doc.corpus <- tm_map(doc.corpus, stripWhitespace)
#force everything back to plaintext document

#Adjust this part!

doc.corpus$content

uniGramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
uniGramMatrix <- TermDocumentMatrix(doc.corpus, control = list(tokenize = uniGramTokenizer))

UnifreqTerms <- findFreqTerms(uniGramMatrix, lowfreq = 5)

UnifreqTerms

sbo_airbnb_dict = sbo_dictionary(pre_data_en, max_size=1000)

summary(sbo_airbnb_dict)

tt_number = as.integer(length(pre_data_en)*0.2)

traindata = pre_data_en[tt_number:length(pre_data_en)]
testdata = pre_data_en[1:tt_number]

p <- sbo_predictor(traindata, # 50k tweets, example dataset
                   N = 3, # Train a 3-gram model
                   dict = sbo_airbnb_dict, # Top 1k words appearing in corpus
                   .preprocess = sbo::preprocess, # Preprocessing transformation
                   EOS = ".?!:;" # End-Of-Sentence characters
)

predict(p, "I am")

predict(p, "The host was the")
predict(p, "I hated about Madrid that")

set.seed(840)
evaluation <- eval_sbo_predictor(p, test = testdata )

evaluation %>% summarise(accuracy = sum(correct)/n(), 
                         uncertainty = sqrt(accuracy * (1 - accuracy) / n())
)

#cleaned_text <- filter(str_detect(pre_data_en, "^[^>]+[A-Za-z\\d]") | pre_data_en !="") 

text = paste(unlist(doc.corpus$content[1:length(doc.corpus$content)]), collapse="\n")[1]

text_1 <- gsub("[\n]{1,}", " ", text)

text_1
#Basic checks from HandsOn 1/3
text[!utf8_valid(text)]
#character(0) -> valid utf-8
text_NFC = utf8_normalize(text)
sum(text_NFC != text)
# Result 0 -> valid 


