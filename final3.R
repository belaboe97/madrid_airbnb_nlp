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
                   filtered = c("<UNK>","EOS>")
)

evaluation <- eval_sbo_predictor(p, test = testdata )

evaluation %>% summarise(accuracy = sum(correct)/n(), 
                         uncertainty = sqrt(accuracy * (1 - accuracy) / n())
)

freq_1 = kgram_freqs(text_1, 1, sbo_airbnb_dict, .preprocess = identity, EOS = "")


doc.corpus<- tm_map(doc.corpus, tolower)

doc.corpus<- tm_map(doc.corpus, removeNumbers)

doc.corpus <- tm_map(doc.corpus, stripWhitespace)

text = paste(unlist(doc.corpus$content[1:length(doc.corpus$content)]), collapse="\n")[1]

text_1 <- gsub("[\n]{1,}", " ", text)



#Markov Chain
words = tokenize_words(text_1)

typeof(words)




predict_word_mchain = function(input){
  return(markovchainSequence(n = 3, 
                      markovchain = fit_markov$estimate,
                      t0 = input , include.t0 = T)) 
}

predict_word_mchain("was")

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





reticulate::use_python('C:/tools/Anaconda3/python.exe',required=T)
reticulate::py_config()
reticulate::py_install('gensim', pip = TRUE)
gensim = reticulate::import('gensim')

gensim$models$KeyedVectors$load_word2vec_format

model = gensim$models$KeyedVectors$load_word2vec_format("C:/Users/Bela Boente/Desktop/Programming/NLP/GoogleNews-vectors-negative300.bin", binary=T)

model$similarity("the","is")

model$most_similar("king")[1:2]


length(outputarray)

outputarray = list()

start_helper = function (){
  
  i = 1
  outputarray = list()
  
  
  while(T){
    
    response = ""
    if(length(outputarray)==0){
      
      val = readline(prompt="Please enter a starting word or sentence: ")
      outputarray[i] <- val
    }
    last_added = tail(outputarray, n=1)
    last_word = stri_extract_last_words(last_added)
    
    labels = c()
    custom_inputs = c("<EOS>","Enter text")
    mchain_prediction = c()
    mchain_prediction_sent = c()
    labels[1] = "End of sentence"
    labels[2] = "Custom input"
    labels[3:5] = "sbo"
    
    unlist(outputarray) %>% paste(collapse = " ") %>% str_to_sentence() -> review
    
    sbo_prediction = predict(p,last_word)
    
    tryCatch({
      
      mchain_prediction = predict_word_mchain(last_word)[2:4]
      mchain_prediction_sent = predict_word_mchain_seq(last_word)[1:2]
      
      mchain_prediction_sent = list(word(mchain_prediction_sent[1], 2:4) %>% paste(collapse = " ") %>% str_to_sentence(), 
                                    word(mchain_prediction_sent[2], 2:4) %>% paste(collapse = " ") %>% str_to_sentence())

      prediction_array = c(sbo_prediction,mchain_prediction,mchain_prediction_sent)
      labels[6:10] = "markov" 
      prediction_array = c(custom_inputs,sbo_prediction,mchain_prediction,mchain_prediction_sent)
      pred_df = as.data.frame(list(predictions = unlist(prediction_array), labels = labels))
    },
    
    error=function(err){
      #print(err)
      #print("No markov chain result found")
      labels = c()
      labels[3:5] = "sbo"
      prediction_array = c(custom_inputs,sbo_prediction)
      pred_df <<- as.data.frame(list(predictions = unlist(prediction_array), labels = labels))
    })
    
    print(review)
    
    print(pred_df)
    response = readline(prompt="Choose a prediction by number: ")  
    
    if(!is.na(as.numeric(response)) && as.numeric(response) != 2){
      outputarray[i+1] = pred_df$predictions[as.numeric(response)]
      similarity_score  = get_similiarity(pred_df$predictions[as.numeric(response)],pred_df)
      print(similarity_score)
    }
    
    if(!is.na(as.numeric(response)) && as.numeric(response) == 2){
      outputarray[i+1] = readline("enter text: ")  
    }
    
    unlist(outputarray) %>% paste(collapse = " ") %>% str_to_sentence() -> review
    
    if(response =="stop"){ 
       print("This is the final result:")
        print(review)
        break 
     }
    
    i = i + 1
  }
  
}

get_similiarity = function(word,listarr){
  
  listarr=listarr$predictions[3:length(listarr$predictions)]
  lword = stri_extract_last_words(word) 
  sim_arr = list()  
  
  for(i in listarr){
    last_word = stri_extract_last_words(i)
    tryCatch({
      sim_arr[last_word] =  model$similarity(lword,last_word)
    }, 
    error = function(err){
      print(i)
    })
    
  }
  
  return(data.frame(sim_arr))
  
}


start_helper()



