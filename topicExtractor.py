from config import OpenAIBot, LangChainBot
import sys
import time
from utils import dataLoader, dataSaver, printAndLog, getBinCount
from bertopic import BERTopic
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from langchain_community.callbacks import get_openai_callback
import openai


class TopicModeller:
    """
    Class for topic modeling using BERTopic.

    Args:
        config (object): Configuration object.
        load (bool, optional): Whether to load a pre-trained topic model. Defaults to False.

    Attributes:
        config (object): Configuration object.
        representationModel (object): Representation model for topic modeling.
        topicModel (object): Topic model.
        topicsOverTime (object): Topics over time.

    Methods:
        loadTopicModel(): Load a pre-trained topic model.
        saveTopicModel(): Save the topic model.
        initializeRepresentationModel(): Initialize the representation model.
        initializeTopicModel(vectorizerModel=None): Initialize the topic model.
        fitTopicModel(combinedTranscript): Fit the topic model.

    """

    def __init__(self, config, load=False):
        self.config = config
        self.representationModel = None
        self.topicModel = None
        self.topicsOverTime = None

        if self.config.callMaxLimit is not None:
            self.callMaxLimit = self.config.callMaxLimit
        else:
            self.callMaxLimit = 5

        if load:
            self.loadTopicModel()
        else:
            self.initializeRepresentationModel()

    def loadTopicModel(self):
        """
        Load a pre-trained topic model.
        """
        self.topicModel = dataLoader(self.config, "topicModel", self.config.videoToUse)
        self.topicsOverTime = dataLoader(
            self.config, "topicsOverTime", self.config.videoToUse
        )

    def saveTopicModel(self):
        """
        Save the topic model.
        """
        dataSaver(
            self.topicModel,
            self.config,
            "topicModel",
            self.config.videoToUse,
        )
        dataSaver(
            self.topicsOverTime,
            self.config,
            "topicsOverTime",
            self.config.videoToUse,
        )

    def initializeRepresentationModel(self):
        """
        Initialize the representation model based on the configuration.
        """
        if self.config.representationModelType == "simple":
            from bertopic.representation import KeyBERTInspired

            self.representationModel = KeyBERTInspired()

        if self.config.representationModelType == "openai":
            from bertopic.representation import OpenAI
            import tiktoken

            OpenAIChatBot = OpenAIBot(self.config)
            tokenizer = tiktoken.encoding_for_model(OpenAIChatBot.model)
            self.representationModel = OpenAI(
                OpenAIChatBot.client,
                model=OpenAIChatBot.model,
                delay_in_seconds=2,
                chat=True,
                nr_docs=8,
                doc_length=None,
                tokenizer=tokenizer,
            )

        if self.config.representationModelType == "langchain":
            from bertopic.representation import LangChain

            LangChainQABot = LangChainBot(self.config)
            self.representationModel = LangChain(
                LangChainQABot.chain, prompt=LangChainQABot.prompt
            )

    def initializeTopicModel(self, vectorizerModel=None):
        """
        Initialize the topic model.

        Args:
            vectorizerModel (object, optional): Vectorizer model for the topic model. Defaults to None.
        """
        if vectorizerModel is not None:
            self.topicModel = BERTopic(
                representation_model=self.representationModel,
                vectorizer_model=vectorizerModel,
            )
        else:
            self.topicModel = BERTopic(representation_model=self.representationModel)

    def fitTopicModel(self, combinedTranscript):
        """
        Fit the topic model.

        Args:
            combinedTranscript (object): Combined transcript object.

        Returns:
            bool: True if successful, False otherwise.
        """
        docs = combinedTranscript["Combined Lines"].tolist()
        timestamps = combinedTranscript["Start"].tolist()

        callAttemptCount = 0
        while callAttemptCount < self.callMaxLimit:
            try:
                topics, probs = self.topicModel.fit_transform(docs)
                binCount = getBinCount(combinedTranscript, windowSize=120)
                self.topicsOverTime = self.topicModel.topics_over_time(
                    docs, timestamps, nr_bins=binCount
                )

                return True

            except openai.AuthenticationError as e:
                printAndLog(f"Error Message: {e}")
                return False
            except openai.RateLimitError as e:
                printAndLog(f"Error Message: {e}")
                printAndLog("Rate limit hit. Pausing for a minute.")
                time.sleep(60)
            except openai.Timeout as e:
                printAndLog(f"Error Message: {e}")
                printAndLog("Timed out. Pausing for a minute.")
                time.sleep(60)
            except Exception as e:
                printAndLog(f"Error Message: {e}")
                printAndLog("Failed to send message.")
                return False
            callAttemptCount += 1

        if callAttemptCount >= self.callMaxLimit:
            printAndLog(
                f"Failed to send message at max limit of {self.callMaxLimit} times."
            )
            return False


def retrieveTopics(config, videoData, overwrite=False):
    """
    Retrieves topics from the given video data using the specified configuration.

    Args:
        config (dict): The configuration settings for topic extraction.
        videoData (list): The video data used for topic extraction.
        overwrite (bool, optional): Whether to overwrite existing topic model and data. Defaults to False.
        saveNameAppend (str, optional): An additional string to append to the saved topic model file name. Defaults to "".

    Returns:
        TopicModeller: The topic modeller object containing the generated topic model and data.
    """
    if not overwrite:
        topicModeller = TopicModeller(config, load=True)
        if (
            topicModeller.topicModel is not None
            and topicModeller.topicsOverTime is not None
        ):
            printAndLog("Topic Model and Data loaded from saved files.")
            printAndLog(
                f"Topics over Time Head: {topicModeller.topicsOverTime.head(5)}",
                logOnly=True,
            )
            return topicModeller

    printAndLog("Generating & saving Topic Model and Data...")

    topicModeller = getTopicsOverTime(config, videoData)
    topicModeller.saveTopicModel()

    printAndLog(f"Topic Model and Data generated and saved for current configuration.")
    printAndLog(
        f"Topics over Time Head: {topicModeller.topicsOverTime.head(5)}", logOnly=True
    )
    return topicModeller


def getVectorizer(combinedTranscript):
    """
    Returns a vectorizer model for extracting features from text data.

    Parameters:
    combinedTranscript (pandas.DataFrame): A DataFrame containing the combined transcript lines.

    Returns:
    CountVectorizer: A vectorizer model for feature extraction.
    """
    import numpy

    numpy.seterr(divide="ignore")

    docs = combinedTranscript["Combined Lines"].tolist()
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 2))

    vocabulary = [k[0] for keyword in keywords for k in keyword]
    vocabulary = list(set(vocabulary))

    vectorizer_model = CountVectorizer(vocabulary=vocabulary)
    return vectorizer_model


def getTopicsOverTime(config, videoData):
    topicModeller = TopicModeller(config)

    vectorizerModel = (
        getVectorizer(videoData.combinedTranscript) if config.useKeyBERT else None
    )
    topicModeller.initializeTopicModel(vectorizerModel)

    # This context manager allows for tracking token usage in the LangChain calls made by BERTopic.
    # THe implementation is a little janky still, and I will need to see how to improve on it.
    with get_openai_callback() as cb:
        fitSuccess = topicModeller.fitTopicModel(videoData.combinedTranscript)

    if fitSuccess and config.representationModelType == "langchain":
        config.topicTokenCount = cb.total_tokens
    else:
        printAndLog("Failed to fit the topic model. Exiting...", level="error")
        sys.exit("Failed to fit the topic model. Exiting...")

    return topicModeller
