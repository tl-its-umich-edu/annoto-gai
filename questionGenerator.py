import pandas as pd
from utils import dataLoader, dataSaver, printAndLog
from config import OpenAIBot
import json


class QuestionData:
    """
    Represents a class that handles question data generation.

    Args:
        config (Config): The configuration object.
        videoData (VideoData): The video data object.
        topicModeller (TopicModeller): The topic modeller object.
        load (bool, optional): Whether to load existing question data. Defaults to True.
    """

    def __init__(self, config, videoData, topicModeller, load=True):
        self.config = config

        self.clusteredTopics = getClusteredTopics(videoData, topicModeller)
        self.dominantTopics = getDominantTopic(self.clusteredTopics)
        self.relevantText, self.questionText = getTextAndQuestion(
            self.dominantTopics, videoData
        )
        self.responseData = None

        if load:
            self.loadQuestionData()
        else:
            self.OpenAIChatBot = OpenAIBot(self.config)
            self.getQuestionData()
            config.questionTokenCount = self.OpenAIChatBot.tokenUsage
            self.saveQuestionData()

    def loadQuestionData(self):
        """
        Loads the question data from a file.
        """
        self.responseData = dataLoader(
            self.config, "questionData", self.config.videoToUse
        )

    def saveQuestionData(self):
        """
        Saves the question data to a file.
        """
        dataSaver(
            self.responseData,
            self.config,
            "questionData",
            self.config.videoToUse,
        )

    def getQuestionData(self):
        """
        Generates the question data using the OpenAI chatbot.
        """
        self.responseData = {}
        for topic in self.questionText:
            response = self.OpenAIChatBot.getResponse(
                self.questionText[topic]
            )
            self.responseData[topic] = response[0]

    def printQuestions(self):
        """
        Prints the question data.
        """
        for topic in self.responseData:
            response = self.responseData[topic]
            response = response.strip('` \n')
            if response.startswith('json'):
                response = response[4:] 
            parsedResponse = json.loads(response)

            printAndLog(f"Topic: {topic}")
            printAndLog(f"Insert Point: {self.dominantTopics[topic]['End'].strftime('%H:%M:%S')}")
            question = f"Question: {parsedResponse['question']}"
            answers = f"Answers: {'\n\t '.join(parsedResponse['answers'])}"
            correct = f"Correct: {parsedResponse['correct']}"
            reason = f"Reason: {parsedResponse['reason']}"
            printAndLog('\n'.join([question, answers, correct, reason]))
            printAndLog('------------------------------------')

# Using a manual overwrite option for debugging.
def retrieveQuestions(config, videoData, topicModeller, overwrite=False):
    """
    Retrieves or generates question data based on the given configuration, video data, and topic modeller.

    Args:
        config (Config): The configuration object.
        videoData (VideoData): The video data object.
        topicModeller (TopicModeller): The topic modeller object.
        overwrite (bool, optional): Whether to overwrite existing question data. Defaults to False.

    Returns:
        QuestionData: The question data object.
    """
    if not config.overwriteQuestionData and not overwrite:
        questionData = QuestionData(config, videoData, topicModeller, load=True)

        if questionData.responseData is not None:
            printAndLog("Question Data loaded from saved files.")
            printAndLog(
                f"Question Data Count: {len(questionData.responseData)}",
                logOnly=True,
            )
            return questionData

    printAndLog("Generating Question Data...", logOnly=True)

    questionData = QuestionData(config, videoData, topicModeller, load=False)
    questionData.saveQuestionData()

    printAndLog("Question Data generated and saved for current configuration.")
    printAndLog(
        f"Question Data Count: {len(questionData.responseData)}",
        logOnly=True,
    )
    return questionData


def getClusteredTopics(videoData, topicModeller):
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

    filteredGroupsDF = pd.concat(filteredGroups)

    # Clean up the words column.
    filteredGroupsDF["Words"] = filteredGroupsDF["Words"].apply(
        lambda keyPhrase: keyPhrase.rstrip(", ")
    )

    # We also filter out any topics that are not relevant in other segments.
    filteredTopics = filteredGroupsDF[filteredGroupsDF["Topic"] != -1]

    # We can merge any segments where the main topic is the same.
    # This gives the inutition that the same topic is being discussed over a period of time.
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

    # We add the topic title for clarity in the final DataFrame.
    clusteredTopics["Topic Title"] = clusteredTopics["Topic"].apply(
        lambda topic: topicModeller.topicModel.topic_labels_[topic]
        .lstrip("0123456789_., ")
        .rstrip("0123456789_., ")
    )

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
    questionText = {}
    for topic in dominantTopics:
        relevantSentences = videoData.combinedTranscript[
            (videoData.combinedTranscript["Start"] >= dominantTopics[topic]["Start"])
            & (videoData.combinedTranscript["End"] <= dominantTopics[topic]["End"])
        ]
        relevantText[topic] = " ".join(relevantSentences["Combined Lines"].tolist())
        questionText[topic] = questionTaskBuilder(topic, relevantText[topic])

    return relevantText, questionText
