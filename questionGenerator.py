import json
import sys
import logging
import pandas as pd
from utils import dataLoader, dataSaver
from configData import OpenAIBot


class QuestionData:
    """
    Represents a class for managing question data.

    Attributes:
        config (str): The configuration for the question data.
        videoData (object): The video data object.
        topicModeller (object): The topic modeller object.
        clusteredTopics (list): The clustered topics.
        dominantTopics (dict): The dominant topics.
        relevantText (str): The relevant text.
        questionQueryText (dict): The question query text.
        responseData (dict): The response data.
        tokenCount (int): The token count.

    Methods:
        __init__(self, config, load=True): Initializes the QuestionData object.
        initializeQuestionData(self, topicModeller, videoData=None): Initializes the question data.
        makeQuestionData(self, load=True): Makes the question data.
        loadQuestionData(self): Loads the question data from a file.
        saveQuestionData(self): Saves the question data to a file.
        getQuestionDataFromResponse(self): Generates the question data using the OpenAI chatbot.
        printQuestions(self): Prints the question data.
    """

    def __init__(self, config, load=True):
        """
        Initializes the QuestionData object.

        Args:
            config (str): The configuration for the question data.
            load (bool, optional): Whether to load the question data. Defaults to True.
        """
        self.config = config
        self.videoData = None

        self.clusteredTopics = None
        self.dominantTopics = None
        self.relevantText = None

        self.questionQueryText = None
        self.responseData = None
        self.tokenCount = 0

    def initialize(self, topicModeller, videoData=None):
        """
        Initializes the question data.

        Args:
            topicModeller (object): The topic modeller object.
            videoData (object, optional): The video data object. Defaults to None.
        """
        # videoData is optional here because we can use the videoData from the topicModeller if it is not provided.
        if videoData is not None:
            self.videoData = videoData
        else:
            self.videoData = topicModeller.videoData

        self.clusteredTopics = getClusteredTopics(topicModeller, videoData)
        self.dominantTopics = getDominantTopic(self.clusteredTopics)
        self.relevantText, self.questionQueryText = getTextAndQuestion(
            self.dominantTopics, videoData
        )

    def makeQuestionData(self, load=True):
        """
        Makes the question data.

        Args:
            load (bool, optional): Whether to load the question data. Defaults to True.
        """
        if load:
            self.loadQuestionData()
        else:
            self.OpenAIChatBot = OpenAIBot(self.config)
            self.getQuestionDataFromResponse()

            self.tokenCount = self.OpenAIChatBot.tokenUsage
            logging.info(
                f"Question Generation Token Count: {self.tokenCount}",
            )

            self.saveQuestionData()

    def loadQuestionData(self):
        """
        Loads the question data from a file.
        """
        loadedData = dataLoader(self.config, "questionData")
        if loadedData is None:
            loadedData = [None] * 5
        if len(loadedData) != 5:
            logging.warning(
                "Loaded data for Question Data is incomplete/broken. Data will be regenerated and saved."
            )
            loadedData = [None] * 5
        (
            self.clusteredTopics,
            self.dominantTopics,
            self.relevantText,
            self.questionQueryText,
            self.responseData,
        ) = loadedData

    def saveQuestionData(self):
        """
        Saves the question data to a file.
        """
        dataSaver(
            (
                self.clusteredTopics,
                self.dominantTopics,
                self.relevantText,
                self.questionQueryText,
                self.responseData,
            ),
            self.config,
            "questionData",
        )

    def getQuestionDataFromResponse(self):
        """
        Generates the question data using the OpenAI chatbot.

        This is because this is a much simpler task than the Topic Extraction part.
        Using LangChain in BERTopic lets BERTopic handle a lot of the complexities in how the topics are generated by leveraging LangChain.
        THat process involves passing keywords, relevant documents, and such to get the human-interpretable topics.

        In the case of Question generation, we only need to send one query per question we need to generate.
        We can get much finer control on the prompt directly because of the simpler query call made.

        LangChain is just a complicated way of building the prompt and storing information at its fundamental level as well, it just hides it from us through abstraction.
        This is a bot of an oversimplification, but it's a good way to think about it.
        So in this case, it made sense to stick to an OpenAIBot for our call rather than using LangChain.

        https://www.singlestore.com/blog/beginners-guide-to-langchain/
        """
        self.responseData = {}
        for topic in self.questionQueryText:
            response = self.OpenAIChatBot.getResponse(self.questionQueryText[topic])
            self.responseData[topic] = response[0]

    def printQuestions(self):
        """
        Prints the question data.
        """
        for topic in self.responseData:
            response = self.responseData[topic]
            response = response.strip("` \n")
            if response.startswith("json"):
                response = response[4:]

            try:
                parsedResponse = json.loads(response)
            except json.JSONDecodeError as e:
                logging.warn(
                    f"Error decoding JSON: {e} for received response: {parsedResponse}"
                )
                logging.warn(f"Question for topic: {topic} not generated.")
                continue

            logging.info(f"Topic: {topic}")
            logging.info(
                f"Insert Point: {self.dominantTopics[topic]['End'].strftime('%H:%M:%S')}"
            )
            question = f"Question: {parsedResponse['question'][:100]+'...'}"
            answers = "Answers: " + "\n\t".join(parsedResponse["answers"])
            correct = f"Correct: {parsedResponse['correct']}"
            reason = f"Reason: {parsedResponse['reason'][:100]+'...'}"
            logging.info("\n".join([question, answers, correct, reason]))

    def printTokenCount(self):
        """
        Prints the token count.
        """
        logging.info(
            f"Question Generation Token Count: {self.tokenCount}",
        )


# Using a manual overwrite option for debugging.
def retrieveQuestions(config, topicModeller=None, videoData=None, overwrite=False):
    """
    Retrieves or generates question data based on the given configuration and topic modeller.

    Args:
        config (Config): The configuration object containing the settings for question generation.
        topicModeller (TopicModeller): The topic modeller object used for topic modeling.
        videoData (VideoData, optional): The video data object containing information about the video. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing question data. Defaults to False.
        Note that overwrite is used only for debugging purposes and should not be set to True in production.
        Use the OVERWRITE_EXISTING_QUESTIONS parameter to overwrite data.

    Returns:
        QuestionData: The question data object containing the generated or retrieved question data.
    """
    questionData = QuestionData(config)
    if not config.overwriteQuestionData and not overwrite:
        questionData.makeQuestionData(load=True)

        if questionData.responseData is not None:
            logging.info("Question Data loaded from saved files.")
            logging.info(f"Question Data Count: {len(questionData.responseData)}")
            return questionData

    if topicModeller is None:
        logging.error(
            "No saved data was found, and Topic Modeller not provided in function call needed to generate Question Data."
        )
        sys.exit("Topic Modeller not provided. Exiting...")
    else:
        questionData.initialize(topicModeller, videoData)

    logging.info("Generating Question Data...")
    questionData.makeQuestionData(load=False)
    questionData.saveQuestionData()

    logging.info("Question Data generated and saved for current configuration.")
    logging.info(f"Question Data Count: {len(questionData.responseData)}")
    return questionData


def getClusteredTopics(topicModeller, videoData=None):
    """
    Get clustered topics from videoData using topicModeller.

    Args:
        videoData (object): The video data object.
        topicModeller (object): The topic modeller object.

    Returns:
        pandas.DataFrame: A DataFrame containing the clustered topics with the following columns:
            - Topic: The dominant topic ID.
            - Words: The words associated with the topic.
            - Frequency: The frequency of the topic.
            - Timestamp: The timestamp of the topic.
            - Start: The start time of the topic segment.
            - End: The end time of the topic segment.
            - Topic Title: The title of the topic.

    """
    if videoData is None:
        videoData = topicModeller.videoData

    # Sort the topic by timsetamp and frequency.
    sortedTopics = topicModeller.topicsOverTime.sort_values(
        ["Timestamp", "Frequency"], ascending=False
    )

    # Remove any segments where the topic is not relevant.
    # These are segments where the topic ID is -1.
    filteredGroups = []
    for group in sortedTopics.groupby("Timestamp"):
        dominantTopic = group[1].iloc[0]["Topic"]
        if dominantTopic != -1:
            filteredGroups.append(group[1])

    if len(filteredGroups) == 0:
        logging.error(
            "Relevant topics not found. This could happen if video is too short or BERTopic could not find any topics."
        )
        sys.exit("No relevant topics found. Exiting...")

    filteredGroupsDF = pd.concat(filteredGroups)

    # Clean up the words column.
    filteredGroupsDF["Words"] = filteredGroupsDF["Words"].apply(
        lambda keyPhrase: keyPhrase.rstrip(", ")
    )

    # We also filter out any topics that are not relevant in other segments.
    filteredTopics = filteredGroupsDF[filteredGroupsDF["Topic"] != -1]

    try:

        # We can merge any segments where the main topic is the same.
        # This gives the intuition that the same topic is being discussed over a period of time.
        clusteredTopics = (
            filteredTopics.groupby(
                (filteredTopics["Topic"] != filteredTopics["Topic"].shift()).cumsum()
            )
            .agg(
                {
                    "Topic": "first",
                    "Words": lambda words: ", ".join(words),
                    "Frequency": "sum",
                    "Timestamp": "first",
                }
            )
            .reset_index(drop=True)
        )

        # We set the start and end times for each topic segment.
        # As segments are linear, the start time is the timestamp of the first topic in the segment, and the end time is the timestamp of the next topic in the segment.
        # This tracks if multiple topics are covered in a single segment.
        clusteredTopics["Start"] = clusteredTopics["Timestamp"]
        clusteredTopics.at[clusteredTopics.index[0], "Start"] = (
            videoData.combinedTranscript["Start"].iloc[0]
        )

        endTimestamps = list(clusteredTopics["Timestamp"].unique()) + [
            videoData.combinedTranscript["End"].iloc[-1]
        ]
        clusteredTopics["End"] = clusteredTopics["Timestamp"].apply(
            lambda timestamp: endTimestamps[endTimestamps.index(timestamp) + 1]
        )

    except Exception as e:
        logging.error(f"Error clustering topics: {e}")
        logging.error(f"Clustering error on Video: {videoData.config.videoToUse}")
        sys.exit(f"Topic clustering error. Exiting...")

    # We add the topic title for clarity in the final DataFrame.
    clusteredTopics["Topic Title"] = clusteredTopics["Topic"].apply(
        lambda topic: topicModeller.topicModel.topic_labels_[topic]
        .lstrip("0123456789_., ")
        .rstrip("0123456789_., ")
    )

    if clusteredTopics.shape[0] == 0:
        logging.info(
            "No clustered topics found. This error should be unlikely and caused by a bug."
        )
        sys.exit("Topic clustering error. Exiting...")

    logging.info(f"Clustered Topics data shape: {clusteredTopics.shape}")
    logging.info(f"Clustered Topics head: {clusteredTopics.head(5)}")

    return clusteredTopics


def getDominantTopic(clusteredTopics):
    """
    Returns the dominant topics based on the frequency of occurrence.

    Parameters:
    clusteredTopics (DataFrame): A DataFrame containing clustered topics with columns "Topic" and "Frequency".

    Returns:
    dominantTopics (DataFrame): A DataFrame containing the dominant topics sorted by start time.
    """
    dominantTopics = []
    for group in clusteredTopics.groupby("Topic"):
        dominantTopics.append(
            group[1].sort_values("Frequency", ascending=False).head(1)
        )

    dominantTopics = pd.concat(dominantTopics).sort_values("Start")

    if dominantTopics.shape[0] == 0:
        logging.error(
            "No dominant topics found. This error should be unlikely and caused by a bug."
        )
        sys.exit("No dominant topics found. Exiting...")

    logging.info(f"Dominant Topics data shape: {dominantTopics.shape}")
    logging.info(f"Dominant Topics head: {dominantTopics.head(5)}")

    return dominantTopics.set_index("Topic Title").to_dict(orient="index")


def questionTaskBuilder(topic, relevantText):
    """
    Generate a multiple choice type question based on the given topic and relevant text.

    Args:
        topic (str): The topic for the question.
        relevantText (str): The relevant text to base the question on.

    Returns:
        str: A formatted string representing the generated question task in JSON format.

    Example:
        For the topic: "Geography", generate a multiple choice type question based on the following transcribed text: "The capital of France is Paris."
        There should be four possible answers, with only one being the correct answer. Also provide a short reason for why each answer is correct or incorrect.
        Return the data in the following JSON format as an example: {"question": "What is the capital of France?", "answers": ["Paris", "London", "Berlin", "Madrid"], "correct": "Paris", "reason": "Paris is the capital of France"}
    """
    questionTask = f"""For the topic: {topic}, generate a multiple choice type question based on the following transcribed text: {relevantText}
There should be four possible answers, with one being the correct answers. Also Provide a short reason to why each answer is correct or incorrect.
Return the data in the following JSON format as an example: {{"question": "What is the capital of France?", "answers": ["Paris", "London", "Berlin", "Madrid"], "correct": "Paris", "reason": "Paris is the capital of France"}}"""

    return questionTask


def getTextAndQuestion(dominantTopics, videoData):
    """
    Retrieves relevant text from the combined transcript based on the given maxTopics.

    Args:
        combinedTranscript (DataFrame): The combined transcript containing the relevant sentences.
        maxTopics (dict): A dictionary containing the maxTopics and their corresponding start and end times.

    Returns:
        dict: A modified version of the maxTopics dictionary with the relevant text and question text added.
    """
    relevantText = {}
    questionQueryText = {}
    for topic in dominantTopics:
        relevantSentences = videoData.combinedTranscript[
            (videoData.combinedTranscript["Start"] >= dominantTopics[topic]["Start"])
            & (videoData.combinedTranscript["End"] <= dominantTopics[topic]["End"])
        ]
        relevantText[topic] = " ".join(relevantSentences["Combined Lines"].tolist())
        questionQueryText[topic] = questionTaskBuilder(topic, relevantText[topic])

        logging.info(
            f"Time range for relevant text for Topic - {topic}: {dominantTopics[topic]['Start'].strftime('%H:%M:%S')} to {dominantTopics[topic]['End'].strftime('%H:%M:%S')}"
        )
        # logging.info(
        #     f"Relevant text selected for Topic - {topic}: {relevantText[topic]}"
        # )
        logging.info(
            f"Query to generate question for Topic - {topic} - First 100 Characters: {questionQueryText[topic][:100]}"
        )

    return relevantText, questionQueryText
