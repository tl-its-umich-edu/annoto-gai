# annoto-gai
This is the Github Project to Annoto GAI work.

Install required additonal libraries using requirements.txt.

For each video, create a folder in the Captions Folder, containing the corresponding .srt file within it. The first .srt file found in the subfolder will be used. 
The captionsProcessor notebook reads captions in the .srt file format provided within the subfolders of the Captions folder.

The getCombinedTranscripts function when run on a single video file name will return a dataframe of the segmented transcript, where each line has an approximate duration, 30s in this case. This will be later used for topic extraction and segmentation.