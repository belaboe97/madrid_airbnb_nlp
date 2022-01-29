library(dplyr)
library(superml)
library(utf8)
library(NLP)
library(tm)
library(qdapDictionaries)
library(tidytext)
library(stringi)
#library(devtools)
library(cld2)
library(cld3)
library(spacyr)
library(stringr)
library(RWeka)
library(ggplot2)
library(wordcloud)
library(sbo)
library(tokenizers)
library(markovchain)
library(word2vec)
library(rword2vec)
library("udpipe")


library(reticulate)


data = read.csv("C:/Users/Bela Boente/Desktop/Programming/NLP/archive/reviews_detailed.csv",encoding = "UTF-8")

data = sample_n(data,500)

utf8_encoded = iconv(data$comments, 'UTF-8', "ASCII")

pre_data = na.omit(utf8_encoded)

pre_data_en = pre_data[detect_language(pre_data) == "en" ]

pre_data_en = na.omit(pre_data_en)

doc.vec = VectorSource(pre_data_en)

doc.corpus = VCorpus(doc.vec)


plot_frequencies = function(wordcorpus,words,minfreq){
  
  tokenizer = function(x) NGramTokenizer(x, Weka_control(min = words, max = words))
  matrix = TermDocumentMatrix(wordcorpus, control = list(tokenize = tokenizer))
  freqTerms = findFreqTerms(matrix, lowfreq = minfreq)
  termFrequency = rowSums(as.matrix(matrix[freqTerms,]))
  termFrequency = data.frame(diagram=names(termFrequency), frequency=termFrequency)  
  
  freqGraph <- ggplot(termFrequency, aes(x=reorder(diagram, frequency), y=frequency)) +
    geom_bar(stat = "identity") +  coord_flip() +
    theme(legend.title=element_blank()) +
    xlab("Wordcombinatio") + ylab("Frequency") +
    labs(title = sprintf("Words by Frequency (%s)",words))
  
  return(freqGraph)
  
}



uniGramGraph <- plot_frequencies(doc.corpus,1,50)
uniGramGraph <- plot_frequencies(doc.corpus,2,30)

#Stupid Backoff algorithm


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

doc.corpus<- tm_map(doc.corpus, tolower)

doc.corpus<- tm_map(doc.corpus, removeNumbers)

doc.corpus <- tm_map(doc.corpus, stripWhitespace)

text = paste(unlist(doc.corpus$content[1:length(doc.corpus$content)]), collapse="\n")[1]

text_1 <- gsub("[\n]{1,}", " ", text)

freq_1 = kgram_freqs(text_1, 1, sbo_airbnb_dict, .preprocess = identity, EOS = "")


#Markov Chain
words = tokenize_words(text_1)

typeof(words)

fit_markov <- markovchainFit(words)


predict_word_mchain = function(input){
  return(markovchainSequence(n = 3, 
                      markovchain = fit_markov$estimate,
                      t0 = input , include.t0 = T)) 
}



predict_word_mchain("was")[2:4]

predict_word_mchain_seq = function(input){
  prediction_array = list()
  for (i in 1:2) {
    markovchainSequence(n = 3, 
                        markovchain = fit_markov$estimate,
                        t0 = input, include.t0 = T) %>% 
      
      # joint words
      paste(collapse = " ") %>% 
      
      # create proper sentence form
      str_replace_all(pattern = " ,", replacement = ",") %>% 
      str_replace_all(pattern = " [.]", replacement = ".") %>% 
      str_replace_all(pattern = " [!]", replacement = "!") %>% 
      
      str_to_sentence() -> prediction 
      prediction_array[i] = prediction
  }
  return(prediction_array)
}

predict_word_mchain_seq("was")[1:3]



reticulate::use_python('C:/tools/Anaconda3/python.exe',required=T)
reticulate::py_config()
reticulate::py_install('gensim', pip = TRUE)
gensim = reticulate::import('gensim')

gensim$models$KeyedVectors$load_word2vec_format

model = gensim$models$KeyedVectors$load_word2vec_format("C:/Users/Bela Boente/Desktop/Programming/NLP/GoogleNews-vectors-negative300.bin", binary=T)

model$similarity("the","is")

model$most_similar("king")[1:3]





predict(p,"house")



