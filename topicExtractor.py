import sys
import logging
import time
from sklearn.feature_extraction.text import CountVectorizer
from langchain_community.callbacks import get_openai_callback
import openai
from bertopic import BERTopic
from keybert import KeyBERT
from configData import OpenAIBot, LangChainBot, useKeyBERT, maxSentenceDuration
from utils import dataLoader, dataSaver, getBinCount
from configData import representationModelType

# This is to suppress the divide by zero warning from numpy.
# This is only used during the CountVectorizer call in the vectorizerModel function.
# The warning does not impede the function, and will be more confusing to the user if it is shown.
# Hence the suppression.
from numpy import seterr

seterr(divide="ignore")


class TopicModeller:
    """
    Class for performing topic modeling on video data.

    Args:
        config (object): Configuration object.
        videoData (object, optional): Video data object. Defaults to None.
    """

    def __init__(self, config, videoData=None):
        self.config = config
        self.videoData = videoData
        self.OpenAIChatBot = None
        self.LangChainTopicBot = None
        self.representationModel = None
        self.tokenCount = 0
        self.callMaxLimit = 3

        self.topicModel = None
        self.topics = None
        self.topicsOverTime = None

    def intialize(self, videoData):
        """
        Initialize the video data.

        Args:
            videoData (object): Video data object.
        """
        self.videoData = videoData

    def makeTopicModel(self, load=True):
        """
        Create or load the topic model.

        Args:
            load (bool, optional): Whether to load a pre-trained topic model. Defaults to True.
        """
        if load:
            self.loadTopicModel()
        else:
            self.initializeRepresentationModel()
            self.getTopicsOverTime()

    def loadTopicModel(self):
        """
        Load a pre-trained topic model.
        """
        self.topicModel = dataLoader(self.config, "topicModel")
        self.topicsOverTime = dataLoader(self.config, "topicsOverTime")

    def saveTopicModel(self):
        """
        Save the topic model and topics over time data.
        """
        dataSaver(self.topicModel, self.config, "topicModel")
        dataSaver(self.topicsOverTime, self.config, "topicsOverTime")

    def initializeRepresentationModel(self):
        """
        Initialize the representation model based on the configuration.
        """
        if representationModelType == "simple":
            from bertopic.representation import KeyBERTInspired

            self.representationModel = KeyBERTInspired()

        if representationModelType == "openai":
            from bertopic.representation import OpenAI
            import tiktoken

            self.OpenAIChatBot = OpenAIBot(self.config)
            tokenizer = tiktoken.encoding_for_model(self.OpenAIChatBot.model)
            self.representationModel = OpenAI(
                self.OpenAIChatBot.client,
                model=self.OpenAIChatBot.model,
                delay_in_seconds=2,
                chat=True,
                nr_docs=8,
                doc_length=None,
                tokenizer=tokenizer,
            )

        if representationModelType == "langchain":
            from bertopic.representation import LangChain

            self.LangChainTopicBot = LangChainBot(self.config)
            self.representationModel = LangChain(
                self.LangChainTopicBot.chain, prompt=self.config.langchainPrompt
            )

    def initializeTopicModel(self, vectorizerModel=None, clusterModel=None):
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
        if clusterModel is not None:
            self.topicModel = BERTopic(
                representation_model=self.representationModel,
                vectorizer_model=vectorizerModel,
                hdbscan_model=clusterModel,
            )
        else:
            self.topicModel = BERTopic(representation_model=self.representationModel)

    def fitTopicModel(self):
        """
        Fit the topic model to the video data.
        """
        docs = self.videoData.combinedTranscript["Combined Lines"].tolist()
        timestamps = self.videoData.combinedTranscript["Start"].tolist()
        vectorizerModel = (
            getVectorizer(self.videoData.combinedTranscript) if useKeyBERT else None
        )

        callAttemptCount = 0
        while callAttemptCount < self.callMaxLimit:
            try:
                self.initializeTopicModel(vectorizerModel)
                self.topics, probs = self.topicModel.fit_transform(docs)
                logging.info(f"Topics and probalities extracted from fitted successfully.")

                if set(self.topics) == {-1}:
                    logging.warning(
                        "All topics are -1. Retrying with K-means clustering..."
                    )
                    KMeansAttempt = self.useKMeans(vectorizerModel, docs)

                    if not KMeansAttempt:
                        logging.warn("Retrying without Vectorizer model.")
                        self.useKMeans(None, docs)

                    if not KMeansAttempt:
                        logging.error("Failed to fit the topic model with K-means clustering. This is unexpected behavior. Exiting...")
                        return False

                # We use `maxSentenceDuration` to determine the number of bins.
                # This ensures that bins will always have atleast one sentence in them.
                binCount = getBinCount(
                    self.videoData.combinedTranscript, windowSize=maxSentenceDuration
                )

                self.topicsOverTime = self.topicModel.topics_over_time(
                    docs, timestamps, nr_bins=binCount
                )

                logging.info(f"Topics over time extracted successfully.")
                return True

            except openai.AuthenticationError as e:
                logging.error(f"Error Message: {e}")
                return False
            except openai.RateLimitError as e:
                logging.error(f"Error Message: {e}")
                logging.error("Rate limit hit. Pausing for a minute.")
                time.sleep(60)
            except openai.Timeout as e:
                logging.error(f"Error Message: {e}")
                logging.error("Timed out. Pausing for a minute.")
                time.sleep(60)
            except Exception as e:
                logging.error(f"Error Message: {e}")
                logging.error("Failed to complete fitting operation on BERTopic.")
                return False
            callAttemptCount += 1

        if callAttemptCount >= self.callMaxLimit:
            logging.error(
                f"Failed to send message at max limit of {self.callMaxLimit} times."
            )
            return False
        
    
    def useKMeans(self, vectorizerModel, docs, n_clusters=3):
        """
        Applies K-means clustering algorithm to group the given documents into topics.

        Args:
            vectorizerModel: The vectorizer model used to transform the documents into feature vectors.
            docs: The list of documents to be clustered.
            n_clusters: The number of clusters/topics to be created (default is 3).

        Returns:
            None
        """

        # This import should not occur frequently, so we only call when it is needed.
        from sklearn.cluster import KMeans

        # We use 3 clusters as a default value. 
        # This means the transcript will be grouped into 3 topics.
        clusterModel = KMeans(n_clusters=n_clusters)
        try:
            self.initializeTopicModel(vectorizerModel, clusterModel)
            self.topics, probs = self.topicModel.fit_transform(docs)
            return True
        except:
            logging.warn("Failed to fit the topic model with K-means clustering.")
            return False

    def getTopicsOverTime(self):
        """
        Get the topics over time.
        """
        with get_openai_callback() as cb:
            fitSuccess = self.fitTopicModel()

        if not fitSuccess:
            logging.error(
                "Failed to fit the topic model. PLease check logs for possible errors."
            )
            sys.exit("Topic model fitting failed. Exiting...")

        if representationModelType == "langchain":
            self.tokenCount = cb.total_tokens

        # This does not work for OpenAI as the token count is not available.
        # Might remove this as we default to LangChain.
        if representationModelType == "openai":
            self.tokenCount = self.OpenAIChatBot.tokenUsage

        logging.info(
            f"Topic Extraction Token Count: {self.tokenCount}",
        )

    def printTopics(self):
        """
        Print the topics.
        """
        getAllTopics = self.topicModel.get_topics()
        topicList = {topicID: getAllTopics[topicID][0][0] for topicID in getAllTopics}
        logging.info(topicList)

    def printTokenCount(self):
        """
        Print the token count.
        """
        logging.info(
            f"Topic Modelling Token Count: {self.tokenCount}",
        )


# Using a manual overwrite option for debugging.
def retrieveTopics(config, videoData=None, overwrite=False):
    """
    Retrieves topics using the specified configuration and video data.

    Args:
        config (Config): The configuration object.
        videoData (VideoData): The video data object.
        overwrite (bool, optional): Whether to overwrite existing topic model and data. Defaults to False.
        Note that overwrite is used only for debugging purposes and should not be set to True in production.
        Use the OVERWRITE_EXISTING_TOPICMODEL parameter to overwrite data.

    Returns:
        TopicModeller: The topic modeller object containing the generated topics and data.
    """
    topicModeller = TopicModeller(config)
    if not config.overwriteTopicModel and not overwrite:
        topicModeller.makeTopicModel(load=True)
        if (
            topicModeller.topicModel is not None
            and topicModeller.topicsOverTime is not None
        ):
            logging.info("Topic Model and Data loaded from saved files.")
            logging.info(
                f"Topics over Time Head: {topicModeller.topicsOverTime.head(5)}"
            )
            return topicModeller

    logging.info("Generating & saving Topic Model and Data...")

    if videoData is None:
        logging.error(
            "No saved data was found, and no video data was provided in function call needed to extract topics."
        )
        sys.exit("Video Data not provided. Exiting...")
    else:
        topicModeller.intialize(videoData)
    topicModeller.makeTopicModel(load=False)
    topicModeller.saveTopicModel()

    logging.info(f"Topic Model and Data generated and saved for current configuration.")
    logging.info(f"Topics over Time Head:\n {topicModeller.topicsOverTime.head(3)}")
    return topicModeller


def getVectorizer(combinedTranscript):
    """
    Returns a vectorizer model for extracting features from text data.

    Parameters:
    combinedTranscript (pandas.DataFrame): A DataFrame containing the combined transcript lines.

    Returns:
    CountVectorizer: A vectorizer model for feature extraction.
    """
    docs = combinedTranscript["Combined Lines"].tolist()
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 2))

    vocabulary = [k[0] for keyword in keywords for k in keyword]
    vocabulary = list(set(vocabulary))

    # Link to the function: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # Basic explanation of Count Vectorizer: https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c
    vectorizer_model = CountVectorizer(vocabulary=vocabulary)
    return vectorizer_model
