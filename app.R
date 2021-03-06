#Install Packages

# install.packages("dplyr")
# install.packages(dplyr)
# install.packages(superml)
# install.packages(utf8)
# install.packages(NLP)
# install.packages(tm)
# install.packages(qdapDictionaries)
# install.packages(tidytext)
# install.packages(stringi)
# install.packages(cld2)
# install.packages(cld3)
# install.packages(spacyr)
# install.packages(stringr)
# install.packages(RWeka)
# install.packages(ggplot2)
# install.packages(wordcloud)
# install.packages(sbo)
# install.packages(tokenizers)
# install.packages(markovchain)
# install.packages(word2vec)
# install.packages("udpipe")
# install.packages(reticulate)

## Required packages
library(dplyr)
library(superml)
library(utf8)
library(NLP)
library(tm)
library(qdapDictionaries)
library(tidytext)
library(stringi)
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
library("udpipe")
library(reticulate)


#Path of app.R file 
setwd("")


########Functions
#All this functions are initiated and used throughout the project
#It is important that all steps are executed sequential.

#######Prediction word for sequences 
#Code optained and adjusted from https://rpubs.com/tahirhussa/207317
plot_frequencies = function(wordcorpus,words,minfreq){
  
  tokenizer = function(x) NGramTokenizer(x, Weka_control(min = words, max = words))
  matrix = TermDocumentMatrix(wordcorpus, control = list(tokenize = tokenizer))
  freqTerms = findFreqTerms(matrix, lowfreq = minfreq)
  termFrequency = rowSums(as.matrix(matrix[freqTerms,]))
  termFrequency = data.frame(diagram=names(termFrequency), frequency=termFrequency)  
  
  freqGraph <- ggplot(termFrequency, aes(x=reorder(diagram, frequency), y=frequency)) +
    geom_bar(stat = "identity") +  coord_flip() +
    theme(legend.title=element_blank()) +
    xlab("Wordcombinations") + ylab("Frequency") +
    labs(title = sprintf("Words by Frequency (%s)",words))
  
  return(freqGraph)
  
}
#######Cleaning operations for corpus 
#Code optained and adjusted from https://rpubs.com/tahirhussa/207317
clean_tm_map = function(corp){
  corp = tm_map(corp, tolower)
  corp = tm_map(corp, removeNumbers)
  corp = tm_map(corp, stripWhitespace)
  return(corp)
}

#######Prediction word for single words
# Code copied and adjusted from https://rpubs.com/Argaadya/markov-chain 
# Chapter Text Generation with N-gram
predict_word_mchain = function(input){
  return(markovchainSequence(n = 3, 
                             markovchain = fit_markov$estimate,
                             t0 = input , include.t0 = T)) 
}

#######Prediction word for sequences 
# Code copied and adjusted from https://rpubs.com/Argaadya/markov-chain 
# Chapter Text Generation with N-gram
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
#######Get Similarities between words
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

#######Application Loop for wordprediction
start_helper = function (){
  
  i = 1
  outputarray = list()
  
  
  while(T){
    
    response = ""
    if(length(outputarray)==0){
      
      val = readline(prompt="Please enter a starting word or sentence: ")
      
      if(!is.na(detect_language(val))){
        
        
        if(detect_language(val) == "en"){
          outputarray[i] <- val 
          i = i+1
        }
        else{
          print("Please enter an english word or sentence")
        }
      }
      else{
        print("Check sentence again, or just simply type: 'Our flat' as starting point (works!)")
      }
    }
    
    if(length(outputarray) > 0){
      
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
        #
        #print(err)
        #print("No markov chain result found")
        labels = c()
        labels[3:5] = "sbo"
        prediction_array = c(custom_inputs,sbo_prediction)
        pred_df <<- as.data.frame(list(predictions = unlist(prediction_array), labels = labels))
      })
      print("The Review:")
      print(review)
      print("___________________________________________________")
      print("This are the predictions")
      print(pred_df)
      print("___________________________________________________")
      response = readline(prompt="Choose a prediction by number: ")  
      
      if(is.na(as.numeric(response)) ||  !between(as.numeric(response),1,length(pred_df$predictions))){
        print(sprintf("Please enter a numeric value between 1 and %s to verify your choice", length(pred_df$predictions)))
      }
      
      else if(!is.na(as.numeric(response)) && as.numeric(response) != 2){
        outputarray[i] = pred_df$predictions[as.numeric(response)]
        similarity_score  = get_similiarity(pred_df$predictions[as.numeric(response)],pred_df)
        print("___________________________________________________")
        print(sprintf("This are the similiarities to the chosen (last) word: %s",pred_df$predictions[as.numeric(response)]))
        print(similarity_score)
        print("___________________________________________________")
      }
      
      else if(!is.na(as.numeric(response)) && as.numeric(response) == 2){
        outputarray[i] = readline("enter text: ")
        similarity_score  = get_similiarity(outputarray[i],pred_df)
        print("___________________________________________________")
        print(sprintf("This are the similiarities to the chosen (last) word: %s",outputarray[i]))
        print(similarity_score)
        print("___________________________________________________")
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
}

#######Start of the application / Cleaning and Analytics


data = read.csv("archive/reviews_detailed.csv",encoding = "UTF-8")

data = sample_n(data,500)

utf8_encoded = iconv(data$comments, 'UTF-8', "ASCII")

pre_data = na.omit(utf8_encoded)

pre_data_en = pre_data[detect_language(pre_data) == "en" ]

pre_data_en = na.omit(pre_data_en)

doc.vec = VectorSource(pre_data_en)

doc.corpus = VCorpus(doc.vec)

#######Prediction word for sequences Visual Analytics / Most common word detection on partly cleaned dataset

g1 <- plot_frequencies(doc.corpus,1,50)
print(g1)

g2 <- plot_frequencies(doc.corpus,2,20)
print(g2)


text = clean_tm_map(doc.corpus)


text = paste(unlist(text), collapse="\n")[1]

text_1 <- gsub("[\n]{1,}", " ", text)


wordcloud(text_1, max.words = 500, random.order = T,rot.per=0.15, use.r.layout=FALSE,colors=brewer.pal(6, "Dark2"))

#######Initializing stupid backoff algorithm on partly cleaned dataset

#create dictionary of airbnb reviewn corpus
sbo_airbnb_dict = sbo_dictionary(pre_data_en, max_size=1000)

tt_number = as.integer(length(pre_data_en)*0.2)

traindata = pre_data_en[tt_number:length(pre_data_en)]
testdata = pre_data_en[1:tt_number]

#Fit model on triandata
#Code copied and adjusted: https://cran.r-project.org/web/packages/sbo/vignettes/sbo.html

p <- sbo_predictor(traindata, # 50k tweets, example dataset
                   N = 5, # Train a 3-gram model
                   dict = sbo_airbnb_dict, # Top 1k words appearing in corpus
                   .preprocess = sbo::preprocess, # Preprocessing transformation
                   EOS = ".?!:;", # End-Of-Sentence characters
                   filtered = c("<UNK>","EOS>")
)
#Evaluate model on testdata
evaluation <- eval_sbo_predictor(p, test = testdata )


evaluation %>% filter(true != "<EOS>") %>% summarise(accuracy = sum(correct)/n(), 
                                                     uncertainty = sqrt(accuracy * (1 - accuracy) / n())
)


#Plot diagramm in which parts of the dictionary most successfull predictions came from 
evaluation %>%
  filter(correct, true != "<EOS>") %>%
  select(true) %>%
  transmute(rank = match(true, table = attr(p, "dict"))) %>%
  ggplot(aes(x = rank)) + geom_histogram(binwidth = 25)



#Refirt model on all data
p <- sbo_predictor(pre_data_en, # 50k tweets, example dataset
                   N = 5, # Train a 3-gram model
                   dict = sbo_airbnb_dict, # Top 1k words appearing in corpus
                   .preprocess = sbo::preprocess, # Preprocessing transformation
                   EOS = ".?!:;", # End-Of-Sentence characters
                   filtered = c("<UNK>","EOS>")
)

#######Initializing Markov Chain on fully cleaned dataset
words = tokenize_words(text_1)
#Fit markov model 
fit_markov <- markovchainFit(words)

#######Import independet source of large model (google) to evaluate similiarities between sbo and markov predictions
## Assumption: Googles Dataset trained on wikipedia, gives a really good estimation if words are similiar or not

#C:/tools/Anaconda3/python.exe
reticulate::use_python('Enter installed python version here, (python where)',required=T)
reticulate::py_config()
reticulate::py_install('gensim', pip = TRUE)

gensim = reticulate::import('gensim')

gensim$models$KeyedVectors$load_word2vec_format

model = gensim$models$KeyedVectors$load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=T)

start_helper()



