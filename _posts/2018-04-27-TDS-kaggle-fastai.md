---
layout: post
title: "A Testament to Power and Simplicity: TDS Rates Top 15 percent in Kaggle Competition, Fast.ai Experiment"
date: 2018-04-27
---
How can a data scientist start using deep learning today and make an immediate impact?  
That was my motivation to enter a worldwide data science competition. I am excited to share a brief description of my experience using fast.ai to attempt this multi-label image classification [Kaggle competition](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018 "iMaterialist Kaggle Competition"). 

With little experience in deep learning, I was able to achieve results that would currently put my submission around 13th place in the competition, or top 15%.  This is no statement about my own abilities, but a testament to the power and simplicity of what is offered by fast.ai.  Unfortunately I missed the initial entry deadline for the competition, but will share the code used [here](https://github.com/dtylor/dtylor.github.io/tree/master/kaggle/imaterialist/).  This post is not meant in any way to be a guide to best practices, but rather another example of how a data scientist can start using deep learning today.

[Fast.ai](http://www.fast.ai/ "Fast.ai") is a wonderful open source resource for practitioners and data scientists to incorporate world class deep learning into their daily analytical challenges. Jeremy Howard and Rachel Thomas have provided an easy to use api and set of straight forward instructions and examples to demystify neural networks and give practical guidance to the everyday user.  Besides providing this helpful and benevolent service, they also donate proceeds to a charity.  Kudos to Jeremy and Rachel for their kind service to the 'little' guy; you don't need to work at Google to solve difficult challenges well. 

I followed fairly closely the code presented in [lesson 3](http://course.fast.ai/lessons/lesson3.html "Fast.ai lesson 3").  The little work required was involved in setting up the GPU environment in [paperspace](http://paperspace.com "paperspace.com") (machine for $0.4/hr) following Jeremy's instructions, downloading the images, creating csv files from the provided jsons for the datasets and running the models. Making the mistake of not sampling and training over all of the data turned out to be a good opportunity to test a background shell script  running overnight (prior to this step, every time the connection to paperspace failed, a restart of jupyter notebook was required).

To download the images, I borrowed from this [script](https://www.kaggle.com/nlecoy/imaterialist-downloader-util?scriptVersionId=3068456 "Script to download images") provided by a fellow kaggler.  His script was edited to increase the maximum number of files downloaded and change the quality of the jpgs from 90% to 100%. Following Jeremy's tip the Chrome extension *curlwget* was used to download the python script directly to the paperspace GPU machine.
The file *download.sh* was created with the following lines to download the images given the urls:
```
	python -u script.py train.json train
	python -u script.py test.json test
	python -u script.py validation.json validation
```
And was run in the background with this command, allowing a check on progress periodically in the output text file *downloadOut*:
```
	nohup sh download.sh > downloadOut &
```
This simple python function was written to parse the json files to a format acceptable to the fast.ai api 
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
Here is the [code](https://github.com/dtylor/dtylor.github.io/tree/master/kaggle/imaterialist/iMaterialist.py "Train Model") used to train the model, incrementing in data size from *64x64* to *128X128* to *300X300*.  The code took about a day to run on the $.4.hr GPU machine in paperspace, run in the background via the file  
*trainModel.sh* with this command
```
	python -u iMaterialist.py
```
and launched with this command
```
	nohup sh trainModel.sh > trainModelOut &
```
It would have been much faster if I had simply sampled to a smaller set.  

The test results of the saved model are reviewed [here](https://github.com/dtylor/dtylor.github.io/tree/master/kaggle/imaterialist/imaterialist_reviewResults.ipynb "Notebook to review model results"). The final f2 metric (I believe slightly different than what was dictated by the competition, but comparable) was *0.5657* which would have put TDS approximately in 13th place at the time of writing this post.
