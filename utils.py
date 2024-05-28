import os
import pickle
import logging
from bertopic import BERTopic
from configData import representationModelType, saveFolder


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
    if config.useKeyBERT:
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
        printAndLog(
            f"Error saving {dataType} for {config.videoToUse}: {e}. Data will need to be reloaded next run.",
            level="warn",
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
    if config.useKeyBERT:
        saveNameAppend = f"_KeyBERT{saveNameAppend}"
    if dataType in ["topicsOverTime", "questionData"]:
        saveNameAppend = f"{saveNameAppend}.p"

    saveName = f"{config.videoToUse}_{representationModelType}{saveNameAppend}"
    savePath = os.path.join(saveFolder, dataType, saveName)

    try:
        if os.path.exists(savePath):
            if dataType == "topicModel":
                return BERTopic.load(savePath)
            return pickle.load(open(savePath, "rb"))
    except Exception as e:
        printAndLog(
            f"Error loading {dataType} for {config.videoToUse}: {e}. Data will need to be reloaded.",
            level="warn",
        )
    return None


def printAndLog(message, level="info", logOnly=False):
    """
    Print a message and log it at the given log level.

    Args:
        message: The message to be printed and logged.
        level: The log level (default is "info").
    """
    if level == "info":
        logging.info(message)
    elif level == "debug":
        logging.debug(message)
    elif level == "warning" or level == "warn":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    elif level == "critical":
        logging.critical(message)
    else:
        raise ValueError(f"Invalid log level: {level}")

    if not logOnly:
        print(message)


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
