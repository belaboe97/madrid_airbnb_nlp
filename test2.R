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



library(tokenizers)
library(markovchain)

words = tokenize_words(text_1)

typeof(words)

fit_markov <- markovchainFit(words)


for (i in 1:2) {
  
  set.seed(i)
  
  markovchainSequence(n = 10, 
                      markovchain = fit_markov$estimate,
                      t0 = "in the apartment", include.t0 = T) %>% 
    
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



library(word2vec)
library(rword2vec)
library("udpipe")


library(reticulate)

reticulate::use_python('C:/tools/Anaconda3/python.exe',required=T)
reticulate::py_config()
reticulate::py_install('gensim', pip = TRUE)
gensim = reticulate::import('gensim')

gensim$models$KeyedVectors$load_word2vec_format

model = gensim$models$KeyedVectors$load_word2vec_format("C:/Users/Bela Boente/Desktop/Programming/NLP/GoogleNews-vectors-negative300.bin", binary=T)

model$similarity("the","is")

model$most_similar("king")[1:3]


