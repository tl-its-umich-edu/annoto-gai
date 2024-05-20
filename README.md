# annoto-gai
This is the Github Project to Annoto GAI work.

## Initial setup: 
1. Install required additonal libraries using `requirements.txt`.  
2. Use the `.env.sample` file in the `config` folder to create an `.env.` file to use with the needed credentials to use UM's OpenAI API.  
3. For each video, create a folder in the `Captions` Folder, containing the corresponding .srt file within it. The first .srt file found in the subfolder will be used.  

You can access videos from [MiVideo](https://www.mivideo.it.umich.edu/). It is recommended to use a video that is over 15 minutes long, and it must have .srt filetype captions available for it. Being familiar with the video contents can help in determining the utility and correctness of the topics extracted.

## Using the notebook: 
The `captionsProcessor.ipynb` notebook reads captions in the .srt file format provided within the subfolders of the `Captions` folder.  

### Processing transcripts: 
The `getCombinedTranscripts` function when run on a single video file name will return a dataframe of the segmented transcript provided within the subfolders of the `Captions` folder, where each line has an approximate duration, 30s in this case. This will be used for topic extraction and segmentation. 

### Extracting topics:
The `retrieveTopics` function takes the segmented transcript and return the `BERTopic` model used, and the topics over time that were extracted from the transcript. Read the notes in the notebook as well for further information about adjusting certain parameters in the `config` Class for managing the choice of Representation model used, and the usage of KeyBERT. 

> Note: Any GenAI-related calls for text generation have been configured to have a temperature of 0 to ensure that the responses received are replicatable and less prone to hallucinations. 

## Using the Python Script:
> Note: Functionality to modify the video transcript name as an Environmental value and within the config will be added in a future commit. 

The `captionsProcessor.py` script currently runs similarly to the `captionsProcessor.ipynb` notebook, but as one continuous script with no visualization options.

## Other notes: 
#### Saving & loading data:
A basic saving and loading functionality is also utilized to load in the model and topics if they have been calculated before. Passing `overwrite=True` to the function will rerun the topic extraction and save an updated version of the data. 

`BERTopic` models cannot be saved as pickle files, and need to used their inbuilt saving mechanism to be saved instead of a pickle. All other data saved is stored as a pickle.

#### Scripts used:
Currently, 3 scripts are used:
1. `utils.py`: Contains common functions used to save and load data, and print and log messages.
2. `transcriptLoader.py`: Contains functions to handle transcript file loading and initital processing.
3. `topicExtractor.py`: Contains functions to handle topic extraction from the processed transcript data.
