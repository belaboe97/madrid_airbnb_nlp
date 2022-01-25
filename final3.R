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

data = read.csv("C:/Users/Bela Boente/Desktop/Programming/NLP/archive/reviews_detailed.csv",encoding = "UTF-8")

utf8_encoded = iconv(data$comments, 'UTF-8', "ASCII")

pre_data = na.omit(utf8_encoded)

pre_data_en = pre_data[detect_language(pre_data) == "en" ]

print(pre_data_en)

pre_data_en = na.omit(pre_data_en)

doc.vec = VectorSource(pre_data_en)
doc.corpus = VCorpus(doc.vec)

doc.corpus<- tm_map(doc.corpus, tolower)

#Remove all punctuatins

#doc.corpus<- tm_map(doc.corpus, removePunctuation)
#Remove all numbers

doc.corpus<- tm_map(doc.corpus, removeNumbers)
#Remove whitespace

doc.corpus <- tm_map(doc.corpus, stripWhitespace)

uniGramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))

uniGramMatrix <- TermDocumentMatrix(doc.corpus, control = list(tokenize = uniGramTokenizer))

UnifreqTerms <- findFreqTerms(uniGramMatrix, lowfreq = 5)

UnitermFrequency <- rowSums(as.matrix(uniGramMatrix[UnifreqTerms,]))
UnitermFrequency <- data.frame(unigram=names(UnitermFrequency), frequency=UnitermFrequency)

g1 <- ggplot(UnitermFrequency, aes(x=reorder(unigram, frequency), y=frequency)) +
  geom_bar(stat = "identity") +  coord_flip() +
  theme(legend.title=element_blank()) +
  xlab("Unigram") + ylab("Frequency") +
  labs(title = "Top Unigrams by Frequency")
print(g1)


sbo_airbnb_dict = sbo_dictionary(pre_data_en, max_size=1000)

tt_number = as.integer(length(pre_data_en)*0.2)

traindata = pre_data_en[tt_number:length(pre_data_en)]
testdata = pre_data_en[1:tt_number]

p <- sbo_predictor(traindata, # 50k tweets, example dataset
                   N = 5, # Train a 3-gram model
                   dict = sbo_airbnb_dict, # Top 1k words appearing in corpus
                   .preprocess = sbo::preprocess, # Preprocessing transformation
                   EOS = ".?!:;", # End-Of-Sentence characters
                   filtered = "<UNK>"
)

evaluation <- eval_sbo_predictor(p, test = testdata )

evaluation %>% summarise(accuracy = sum(correct)/n(), 
                         uncertainty = sqrt(accuracy * (1 - accuracy) / n())
)

babble(p)


