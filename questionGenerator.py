import logging
import sys
from transcriptLoader import retrieveTranscript
from topicExtractor import retrieveTopics
from BERTopicQuestionGenerator import retrieveBERTopicQuestions
from LangChainQuestionGenerator import retrieveLangChainQuestions


def retrieveQuestions(
    config, topicModeller=None, videoData=None, overwrite=False, saveToFile=True
):
    """
    Retrieves questions based on the specified generation model.

    Args:
        config (object): Configuration object containing generation model information.
        topicModeller (object, optional): Topic modeller object. Defaults to None.
        videoData (object, optional): Video data object. Defaults to None.
        overwrite (bool, optional): Flag indicating whether to overwrite existing data. Defaults to False.
        saveToFile (bool, optional): Flag indicating whether to save the question data to a file. Defaults to True.

    Returns:
        object: Question data object.
    """
    if config.generationModel == "BERTopic":
        questionData = retrieveBERTopicQuestions(
            config,
            topicModeller=topicModeller,
            videoData=videoData,
            overwrite=overwrite,
        )
    elif config.generationModel == "LangChain":
        questionData = retrieveLangChainQuestions(
            config, videoData=videoData, overwrite=overwrite
        )
    else:
        logging.error(
            f"Invalid generation model specified: {config.generationModel}, valid options are 'BERTopic' and 'LangChain'."
        )
        sys.exit("Invalid generation model specified. Exiting...")

    if saveToFile:
        questionData.saveToFile()
    return questionData


def processCaptions(config, overwrite=False, saveToFile=True):
    """
    Process captions to retrieve transcript, topics, and questions for a given video.

    Args:
        config (object): Configuration object containing video information.
        overwrite (bool, optional): Flag indicating whether to overwrite existing data. Defaults to False.
        saveToFile (bool, optional): Flag indicating whether to save the questions to a file. Defaults to True.

    Returns:
        None
    """
    logging.info(f"Retrieving Transcript for {config.videoToUse}...")
    videoData = retrieveTranscript(config, overwrite=overwrite)

    topicModeller = None
    if config.generationModel == "BERTopic":
        logging.info(f"Retrieving Topics for {config.videoToUse}...")
        topicModeller = retrieveTopics(config, videoData, overwrite=overwrite)
        topicModeller.printTopics()

    logging.info(f"Retrieving Questions for {config.videoToUse}...")
    questionData = retrieveQuestions(
        config,
        topicModeller=topicModeller,
        videoData=videoData,
        overwrite=overwrite,
        saveToFile=saveToFile,
    )
    questionData.printQuestions()
