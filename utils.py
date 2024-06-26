import os
import pickle
import logging
from bertopic import BERTopic
from typing import List
from langchain_core.documents import Document
from configData import representationModelType, saveFolder, useKeyBERT


def dataSaver(data, config, dataType, saveNameAppend=""):
    """
    Save the data based on the specified configuration.

    Args:
        data: The data to be saved.
        config: The configuration object.
        dataType: The type of data being saved.
        saveNameAppend: An optional string to append to the save name.

    Returns:
        The path where the data is saved.
    """
    if useKeyBERT and config.generationModel == "BERTopic":
        saveNameAppend = f"_KeyBERT{saveNameAppend}"
    saveName = f"{config.videoToUse}_{representationModelType}{saveNameAppend}"
    savePath = os.path.join(saveFolder, dataType, saveName)

    try:
        if dataType == "topicModel":
            data.save(
                savePath,
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )
        else:
            pickle.dump(data, open(savePath + ".p", "wb"))
        return True

    except Exception as e:
        logging.warn(
            f"Error saving {dataType} for {config.videoToUse}: {e}. Data will need to be reloaded next run."
        )
        return False


def dataLoader(config, dataType, saveNameAppend=""):
    """
    Load data based on the specified configuration, data type, video to use, and save name appendix.

    Parameters:
    - config: The configuration object.
    - dataType: The type of data to load.
    - saveNameAppend: An optional appendix to add to the save name.

    Returns:
    - The loaded data if it exists, otherwise False.
    """
    if useKeyBERT and config.generationModel == "BERTopic":
        saveNameAppend = f"_KeyBERT{saveNameAppend}"
    if dataType != "topicModel":
        saveNameAppend = f"{saveNameAppend}.p"

    saveName = f"{config.videoToUse}_{representationModelType}{saveNameAppend}"
    savePath = os.path.join(saveFolder, dataType, saveName)

    try:
        if os.path.exists(savePath):
            if dataType == "topicModel":
                return BERTopic.load(savePath)
            return pickle.load(open(savePath, "rb"))
    except Exception as e:
        logging.warn(
            f"Error loading {dataType} for {config.videoToUse}: {e}. Data will need to be reloaded."
        )
    return None


def getBinCount(combinedTranscript, windowSize=120):
    """
    Calculates the number of bins based on the combined transcript and window size.

    Parameters:
    combinedTranscript (DataFrame): The combined transcript containing the start and end times.
    windowSize (int): The size of each window in seconds. Default is 120.

    Returns:
    int: The number of bins calculated based on the video duration and window size.
    """
    videoDuration = (
        combinedTranscript["End"].iloc[-1] - combinedTranscript["Start"].iloc[0]
    )
    binCount = int(videoDuration.total_seconds() // windowSize)
    return binCount


def formatDocs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Text ID: {doc.metadata['ID']}"
        + f"\nText Start Time: {doc.metadata['Start']}"
        + f"\nText End Time: {doc.metadata['End']}"
        + f"\nText: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


def getMetadata(transcript):
    """
    Converts the timestamps in the transcript dataframe to a specific format and adds an 'ID' column.

    Args:
        transcript (pandas.DataFrame): The transcript dataframe containing 'Start' and 'End' columns.

    Returns:
        pandas.DataFrame: The modified transcript dataframe with converted timestamps and an added 'ID' column.
    """
    for timeCol in ["Start", "End"]:
        transcript[timeCol] = transcript[timeCol].apply(
            lambda timestamp: timestamp.strftime("%H:%M:%S")
        )
    transcript["ID"] = transcript.index

    return transcript
