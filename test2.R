library(text2vec)
library(data.table)
library(magrittr)

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

doc.corpus<- tm_map(doc.corpus, removePunctuation)
#Remove all numbers

doc.corpus<- tm_map(doc.corpus, removeNumbers)
#Remove whitespace

doc.corpus <- tm_map(doc.corpus, stripWhitespace)
#force everything back to plaintext document

text = paste(unlist(doc.corpus$content[1:length(doc.corpus$content)]), collapse="\n")[1]

text_1 <- gsub("[\n]{1,}", " ", text)


library(tokenizers)
library(markovchain)

words = tokenize_words(text_1)

typeof(words)

fit_markov <- markovchainFit(words)


for (i in 1:10) {
  
  set.seed(i)
  
  markovchainSequence(n = 10, 
                      markovchain = fit_markov$estimate,
                      t0 = "apartment", include.t0 = T) %>% 
    
    # joint words
    paste(collapse = " ") %>% 
    
    # create proper sentence form
    str_replace_all(pattern = " ,", replacement = ",") %>% 
    str_replace_all(pattern = " [.]", replacement = ".") %>% 
    str_replace_all(pattern = " [!]", replacement = "!") %>% 
    
    str_to_sentence() %>% 
    
    print()
}


predictive_text <- function(text, num_word){
  text <- strsplit(text, " ") %>% unlist() %>% tail(1)
  
  # exclude punctuation
  punctuation <- which(fit_markov$estimate[ tolower(text), ] %>% names() %>% str_detect("[:punct:]"))
  
  suggest <- fit_markov$estimate[ tolower(text), -punctuation] %>%
    sort(decreasing = T) %>% 
    head(num_word) 
  
  suggest <- suggest[suggest > 0] %>% 
    names()
  
  return(suggest)
}

pdt = predictive_text("i am", 3)
pdt


library(word2vec)
library(rword2vec)
library("udpipe")

writeLines(text_1, "airbnb_model.bin")

read.word2vec()
path <- system.file(package = "word2vec", "models", "C:/Users/Bela Boente/Desktop/Programming/NLP/GoogleNews-vectors-negative300.bin")
model <-read.word2vec("C:/Users/Bela Boente/Desktop/Programming/NLP/GoogleNews-vectors-negative300.bin")
model$vocabulary[1]

library(reticulate)

reticulate::use_python('C:/tools/Anaconda3/python.exe',required=T)
reticulate::py_config()
reticulate::py_install('gensim', pip = TRUE)
gensim = reticulate::import('gensim')

gensim$models$KeyedVectors$load_word2vec_format

model = gensim$models$KeyedVectors$load_word2vec_format("C:/Users/Bela Boente/Desktop/Programming/NLP/GoogleNews-vectors-negative300.bin", binary=T)

model$similarity("the","is")

model$most_similar("king")[1:3]




















word2vec::read.wordvectors("C:/Users/Bela Boente/Desktop/Programming/NLP/GoogleNews-vectors-negative300.bin",)
#model=word2vec(x = doc.corpus, dim = 15, iter = 20)
#dist=distance(model,search_word = "king",num = 10)
embedding <- as.matrix(model)
embedding <- predict(model, c("the", "is"), type = "embedding")
lookslike <- predict(model, c("the", "is"), type = "nearest", top_n = 5)

lookslike

typeof(unlist(doc.corpus))

model <- word2vec(x = text_1, dim = 15, iter = 20)


model <- read.word2vec(file = "cb_ns_500_10.w2v", normalize = TRUE)

emb <- as.matrix(model)

word2vec_similarity(pdt[1], pdt[2])



