from config import OpenAIBot, LangChainBot
from utils import dataLoader, dataSaver
import logging
from bertopic import BERTopic
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from langchain_community.callbacks import get_openai_callback


def retrieveTopics(
    videoToUse, transcriptToUse, config, overwrite=False, saveNameAppend=""
):
    """
    Retrieves or generates topic model and data.

    Args:
        videoToUse (str): Path to the video file.
        transcriptToUse (str): Path to the transcript file.
        config (dict): Configuration settings.
        overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.
        saveNameAppend (str, optional): String to append to the save name. Defaults to "".

    Returns:
        tuple: A tuple containing the topics over time and the topic model.
    """

    if not overwrite and (
        (topicModel := dataLoader(config, "topicModel", videoToUse, saveNameAppend))
        is not False
        and (
            topicsOverTime := dataLoader(
                config, "topicsOverTime", videoToUse, saveNameAppend
            )
        )
        is not False
    ):
        logging.info("Topic Model and Data loaded from saved files.")
        return topicsOverTime, topicModel

    logging.info("Generating & saving Topic Model and Data.")
    topicsOverTime, topicModel = getTopicsOverTime(transcriptToUse, config=config)
    dataSaver(topicModel, config, "topicModel", videoToUse, saveNameAppend)
    dataSaver(
        topicsOverTime,
        config,
        "topicsOverTime",
        videoToUse,
        saveNameAppend,
    )

    return topicsOverTime, topicModel


def getVectorizer(docs):
    """
    Returns a vectorizer model based on the given documents.

    Args:
        docs (list): A list of documents.

    Returns:
        vectorizer_model: A CountVectorizer model configured with the extracted keywords from the documents.
    """

    import numpy

    numpy.seterr(divide="ignore")

    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 2))

    vocabulary = [k[0] for keyword in keywords for k in keyword]
    vocabulary = list(set(vocabulary))

    vectorizer_model = CountVectorizer(vocabulary=vocabulary)
    return vectorizer_model


def getTopicsOverTime(combinedTranscript, config):
    """
    Get topics over time from a combined transcript.

    Args:
        combinedTranscript (DataFrame): The combined transcript containing the lines and timestamps.
        config (Config): The configuration object specifying the representation model type and other settings.

    Returns:
        tuple: A tuple containing the topics over time and the topic model.

    Raises:
        None

    """

    docs = combinedTranscript["Combined Lines"].tolist()
    timestamps = combinedTranscript["Start"].tolist()

    if config.representationModelType == "simple" or config is None:
        from bertopic.representation import KeyBERTInspired

        representation_model = KeyBERTInspired()

    if config.representationModelType == "openai":
        from bertopic.representation import OpenAI
        import tiktoken

        OpenAIChatBot = OpenAIBot(config)
        tokenizer = tiktoken.encoding_for_model(OpenAIChatBot.model)
        representation_model = OpenAI(
            OpenAIChatBot.client,
            model=OpenAIChatBot.model,
            delay_in_seconds=2,
            chat=True,
            nr_docs=8,
            doc_length=None,
            tokenizer=tokenizer,
        )

    if config.representationModelType == "langchain":
        from bertopic.representation import LangChain

        LangChainQABot = LangChainBot(config)
        chain = LangChainQABot.chain
        representation_model = LangChain(chain, prompt=LangChainQABot.prompt)

    if config.useKeyBERT:
        topicModel = BERTopic(
            representation_model=representation_model,
            vectorizer_model=getVectorizer(docs),
        )
    else:
        topicModel = BERTopic(representation_model=representation_model)

    with get_openai_callback() as cb:

        topics, probs = topicModel.fit_transform(docs)

        # hierarchical_topics = topic_model.hierarchical_topics(docs)
        # topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

        binCount = getBinCount(combinedTranscript, windowSize=120)
        topicsOverTime = topicModel.topics_over_time(docs, timestamps, nr_bins=binCount)

    # topicModel.visualize_topics_over_time(topicsOverTime)
    if config.representationModelType == "langchain":
        LangChainQABot.tokenUsage = cb.total_tokens

    config.topicTokenCount = LangChainQABot.tokenUsage

    return topicsOverTime, topicModel


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
