##Important Note: Use RStudio to execute the script sequential. Variables are stored in the Environment and then accessed throughout the script by the defined functions

1) Download the dataset from kaggle: https://www.kaggle.com/rusiano/madrid-airbnb-data
2) Place the folder "archive" in the same folder app.R is in
3) Download the GoogleNews-vectors-negative 300 and unzip 
	A quick guide (https://stackoverflow.com/questions/46433778/import-googlenews-vectors-negative300-bin)
	install wget
	wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
	gzip -d GoogleNews-vectors-negative300.bin.gz

4) Place the unzipped .bin file in the same folder of app.R 
5) Install all requirements (uncomment the install.packages at the start of app.R
6) Set working directory to folder with app.R (app.R - line 51)
7) (Install python) pip install gensim
8) Set python path in app.R -> can be found with "python where" command in cmd

##Important notes on Review Helper Writer

- With Numbers between 1-10 you can specify next action. 
- Enter "stop" to stop the execution and get the result.