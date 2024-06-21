# annoto-gai
This is the Github Project to Annoto GAI work.

## Initial setup: 
1. Install required additonal libraries using `requirements.txt`.  
2. Use the `.env.sample` file in the `config` folder to create an `.env.` file to use with the needed credentials to use UM's OpenAI API.  
3. For each video, create a folder in the `Captions` Folder, containing the corresponding .srt file within it. The first .srt file found in the subfolder will be used.  

You can access videos from [MiVideo](https://www.mivideo.it.umich.edu/). It is recommended to use a video that is over 15 minutes long, and it must have .srt filetype captions available for it. Being familiar with the video contents can help in determining the utility and correctness of the topics extracted.

There are two generation models to generate questions from a transcript:
1. `BERTopic`: This approach first attempts to identify the topics being discussed in the transcript, and relevant text associated with the topics. By first explictly finding the topics covered, the generated questions should pertain more directly to them, and be of more relevance to the content.
2. `LangChain`: This approach leverages a purely GenAI-based system using LangChain. The transcript is segmented and a list of Questions are returned as per the prompt requirment for the type of question. This is a simpler approach, but has a higher chance to generate questions that are of lower quality or relevance.

## Using the notebook: 
The `captionsProcessor.ipynb` notebook reads captions in the .srt file format provided within the subfolders of the `Captions` folder. The notebook mainly serves as a way to test and debug code. Ideally, use the `captionsProcessor.py` script when processing a file.

### Processing transcripts: 
The `retrieveTranscript` function when passed a `configVars` class object  will return a class object containing information for the segmented transcript provided within the subfolders of the `Captions` folder. This will be used for topic extraction and segmentation. 

Refer additional notes below for further details on the segmentation process.

### Extracting topics:
This is only used when using the `BERTopic` question generation model. 

The `retrieveTopics` function takes the segmented transcript and return the `BERTopic` model used, and the topics over time that were extracted from the transcript. 
The `.env` file contains some variables that can be adjusted for this process, but it is advisable to stick to defaults set unless testing. 

> Note: Any GenAI-related calls for text generation have been configured to have a temperature of 0 to ensure that the responses received are replicatable and less prone to hallucinations. 

### Generating Questions:
The `retrieveQuestions` function takes the segmented transcript (and the topic model if required) to produce the generated questions for the given transcript. The `QUESTION_COUNT` variable in the `.env` file sets the number of questions that are to be generated per transcript. Refer to the `.env` file for details on setting a question count.

## Using the Python Script:
The `captionsProcessor.py` script reads the configurations parameters set in the `.env` file to generate question data for the transcript. All question data generated will be saved to a folder labelled `Output Data`, as a `.txt` within a subfolder with the same name as the corresponding captions folder the transcript was loaded from.

## Other notes: 
#### Saving & loading data:
A basic saving and loading functionality is also utilized to load in the model and topics if they have been calculated before. Passing `overwrite=True` to the `retrieveTranscript`, `retrieveTopics`, `retrieveQuestions`, or `processCaptions` functions will rerun them to save an updated version of the data. Ideally, use the `.env` to adjust this setting, and only use `overwrite=True` when debugging. 

`BERTopic` models cannot be saved as pickle files, and need to used their inbuilt saving mechanism to be saved instead of a pickle. All other data saved is stored as a pickle.
This is also why the `TopicModeller` class can't be saved as single entity easily. Saving the model in it needs a different mechanism. 
In a future implementation, the need for the model being saved itself could be removed, but I do not believe this is viable right now.

#### More details on transcript segmentations:
The `SpaCy` library is used to attempt to split the transcript in such a way that whole sentences form each segment, rather than cutting off mid-sentence. By default, each segment will be at least 30s long, going longer till the end of a speaker's sentence. This improves the likelihood that a question is inserted only at the end of a sentence. In some cases, sentence-based segmentation can fail, where sentences appear to be over 120s long in duration. This can happen when trying to use raw YouTube transcriptions, or have a speaker who has a tendency to ramble without pauses. In those cases, sentences are simply segmented by word, where each segment of words has an approximate duration, 30s default in this case.

#### Understanding the generated question data:
Question data is saved to a folder with the same name in an output folder called `Output Data` as the corresponding folder from the `Captions` folder whose transcript was used to generate the data from. The file is called `Questions - {generationModel}.txt` where `generationModel` refers to the type of question generation model used. For a given video, the file will list the Video/Folder Name, the Topic, the timestamps for the selected relevant text (Currently only through `BERTopic`), insertion time for the question, and then the questions itself. The question generated will be a multiple choice type, with 4 questions, one of which is the correct. A reasoning is also provided for this answer to be the correct one. 

#### Scripts used:
Currently, 8 scripts are used:
1. `configData.py`: Contains the `configVars` class used to store environmental variables, as well as two other classes for OpenAI and LangChain API usage.
2. `utils.py`: Contains common functions used to save and load data, as well as some functions for formatting transcript data to feed into the LangChain piepline.
3. `transcriptLoader.py`: Contains the `TranscriptData` class that handles transcript file loading and initital processing.
4. `topicExtractor.py`: Contains the `TopicModeller` class and functions to handle topic extraction from the processed transcript data.
5. `BERTopicQuestionGenerator.py`: Handles question extraction when using the `BERTopic` question generation mode.
6. `LangChainQuestionGenerator.py`: Handles question extraction when using the `LangChain` question generation mode.
7. `questionGenerator.py`: Contains the `retrieveQuestions` and `processCaptions` functions to simplify the calls needed to generate question data. 
8. `captionsProcessor.py`: Simply runs question generation based on parameters set in the `.env` file.

