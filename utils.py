import os
import pickle

from bertopic import BERTopic


def dataSaver(data, config, dataType, videoToUse, saveNameAppend=""):
    """
    Save the data based on the specified configuration.

    Args:
        data: The data to be saved.
        config: The configuration object.
        dataType: The type of data being saved.
        videoToUse: The video identifier.
        saveNameAppend: An optional string to append to the save name.

    Returns:
        The path where the data is saved.
    """
    if config.useKeyBERT:
        saveNameAppend = f"_KeyBERT{saveNameAppend}"
    saveName = f"{videoToUse}_{config.representationModelType}{saveNameAppend}"
    savePath = os.path.join(config.saveFolder, dataType, saveName)

    if dataType == "topicModel":
        data.save(
            savePath,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
    else:
        pickle.dump(data, open(savePath + ".p", "wb"))

    return savePath


def dataLoader(config, dataType, videoToUse, saveNameAppend=""):
    """
    Load data based on the specified configuration, data type, video to use, and save name appendix.

    Parameters:
    - config: The configuration object.
    - dataType: The type of data to load.
    - videoToUse: The video to use for loading the data.
    - saveNameAppend: An optional appendix to add to the save name.

    Returns:
    - The loaded data if it exists, otherwise False.
    """
    if config.useKeyBERT:
        saveNameAppend = f"_KeyBERT{saveNameAppend}"
    if dataType in ["topicsOverTime", "questionData"]:
        saveNameAppend = f"{saveNameAppend}.p"

    saveName = f"{videoToUse}_{config.representationModelType}{saveNameAppend}"
    savePath = os.path.join(config.saveFolder, dataType, saveName)

    if os.path.exists(savePath):
        if dataType == "topicModel":
            return BERTopic.load(savePath)

        return pickle.load(open(savePath, "rb"))

    return False
