---
layout: post
title: "A Testament to Power and Simplicity: TDS Rates Top 15 percent in Kaggle Competition, Fast.ai Experiment"
date: 2018-04-27
---
How can a data scientist start using deep learning today and make an immediate impact?  
That was my motivation to enter a worldwide data science competition. I am excited to share a brief description of my experience using fast.ai to attempt this multi-label image classification Kaggle competition [Kaggle competition](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018 "iMaterialist Kaggle Competition"). 

With little experience in deep learning, I was able to achieve results that would currently put my submission around 13th place in the competition.  This is no statement about my own abilities, but a testament to the power and simplicity of what is offered by fast.ai.  Unfortunately I missed the initial entry deadline for the competition.  I will share the code I used.  This post is not meant in any way to be a guide to best practices, but rather another example of how a data scientist can start using deep learning today.

[Fast.ai](http://www.fast.ai/ "Fast.ai") is a wonderful open source resource for practitioners and data scientists to incorporate world class deep learning into their daily analytical challenges. Jeremy Howard and Rachel Thomas have provided an easy to use api and set of straight forward instructions and examples to demystify neural networks and give practical guidance to the everyday user.  Besides providing this helpful and benevolent service, they also donate proceeds to a charity.  Kudos to Jeremy and Rachel for their kind service to the 'little' guy; you don't need to work at Google to solve difficult challenges well. 

I followed fairly closely the code presented in [lesson 3](http://course.fast.ai/lessons/lesson3.html "Fast.ai lesson 3").  Most of my work was involved in setting up the GPU environment in [paperspace](http://paperspace.com "paperspace.com") (machine for $0.4/hr) following Jeremy's instructions, downloading the images, creating csv files from the provided jsons for training, validation and test datasets and running the models. I made the mistake of not sampling my data and running over all of the data provided, but this turned out to be a good opportunity for me to test running a background shell script overnight to run my python code (prior to this step, every time my connection to paperspace failed I would need to rerun my jupyter notebook).

To download the images, I borrowed from this [script](https://www.kaggle.com/nlecoy/imaterialist-downloader-util?scriptVersionId=3068456 "Script to download images") provided by a fellow kaggler.  I edited his script to increase the maximum number of files downloaded and changed the quality of the jpgs to 100%. Following Jeremy's tip I downloaded the Chrome extension *curlwget* and used it to download the python script directly to the paperspace GPU machine.
To run this code, I created the file *download.sh* to run in the background, even if the connection to paperspace is broken:
```
	python -u script.py train.json train
	python -u script.py test.json test
	python -u script.py validation.json validation
```
And ran it in the background with this command, allowing me to check progress periodically in the output text file *downloadOut*:
```
	nohup sh download.sh > downloadOut &
```
I wrote this simple python function to parse the json files to a format acceptable to the fast.ai api 
```
	def createImageCSV(_dataset):
		fn ='data/imaterialist/'+_dataset+'.json'
		outfn='data/imaterialist/'+_dataset+'.csv'
		f = open(fn, 'r') 
		outf  = open(outfn, 'w') 
		data = json.load(f)
		ano =  data["annotations"]
		for ano in data["annotations"]:
			imageId = ano["imageId"]
			labelId = ano["labelId"]
			outf.write(imageId + ".jpg," + ' '.join(labelId)+'\n')
		f.close
		outf.close
	createImageCSV('train')
	createImageCSV('validation')		
```
In this [jupyter notebook](https://github.com/dtylor/dtylor.github.io/tree/master/kaggle/imaterialist/imaterialist.ipynb "Notebook to assess data"), I assessed a sample image and the distribution of image sizes in the training set.
Here is the [code](https://github.com/dtylor/dtylor.github.io/tree/master/kaggle/imaterialist/iMaterialist.py "Train Model") used to train the model, incrementing in data size from *64x64* to *128X128* to *300X300*.  The code took about a day to run on my $.4.hr GPU machine.  I ran it in the background via the file  
*trainModel.sh*
```
	python -u iMaterialist.py
```
and launched the script with this command
```
	nohup sh trainModel.sh > trainModelOut &
```
It would have been much faster if I had simply sampled to a smaller set.  

The test results of the saved model are reviewed [here](https://github.com/dtylor/dtylor.github.io/tree/master/kaggle/imaterialist/imaterialist_reviewResults.ipynb "Notebook to review model results"). The final f2 metric (I believe slightly different than what was dictated by the competition, but comparable) was *0.5657* which would put me approximately in 13th place at the time of writing this post.