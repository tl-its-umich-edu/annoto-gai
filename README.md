# annoto-gai
This is the Github Project to Annoto GAI work.

Initial setup: 
> Install required additonal libraries using `requirements.txt`.  
> Use the `.env.sample` file in the `config` folder to create an `.env.` file to use with the needed credentials to use UM's OpenAI API.  
> For each video, create a folder in the `Captions` Folder, containing the corresponding .srt file within it. The first .srt file found in the subfolder will be used.  

You can access videos from [MiVideo](https://www.mivideo.it.umich.edu/). It is recommended to use a video that is over 15 minutes long, and it must have .srt filetype captions available for it. Being familiar with the video contents can help in determining the utility and corectness of the topics extracted.

The `captionsProcessor.ipynb` notebook reads captions in the .srt file format provided within the subfolders of the `Captions` folder.