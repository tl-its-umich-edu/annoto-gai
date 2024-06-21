import json
import os
import sys
import logging
import pandas as pd
from utils import dataLoader, dataSaver
from configData import OpenAIBot, outputFolder, minTopicFrequency


class BERTopicQuestionData:
    """
    Represents a class for generating and managing question data.

    Attributes:
        config (str): The configuration for the question data.
        videoData (object): The video data object.
        clusteredTopics (object): The clustered topics.
        questionInfo (object): The question information.
        responseInfo (object): The response information.
        tokenCount (int): The token count for question generation.

    Methods:
        __init__(self, config, load=True): Initializes the QuestionData object.
        initialize(self, topicModeller, videoData=None): Initializes the question data.
        makeQuestionData(self, load=True): Makes the question data.
        loadQuestionData(self): Loads the question data from a file.
        saveQuestionData(self): Saves the question data to a file.
        printQuestions(self): Prints the generated questions, answers, correct answer, and reason for a given response.
        printTokenCount(self): Prints the token count.
        saveToFile(self): Saves the question data to a file.
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
        self.questionInfo = None

        self.responseInfo = None
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

        # We don't track relevantRegions as questionInfo contains the same information.
        relevantRegions = getRelevantRegions(
            self.clusteredTopics, self.config.questionCount, minTopicFrequency
        )

        # Check if the contextWindowSize is set to 0, which means we do not want to truncate the relevant text.
        if self.config.contextWindowSize != 0:
            # Truncate the relevant text as needed based on the contextWindowSize.
            relevantRegions = truncateRelevantText(
                relevantRegions, self.videoData, self.config.contextWindowSize
            )

        self.questionInfo = getTextAndQuery(relevantRegions, self.videoData)

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
            self.responseInfo = getQuestionDataFromResponse(
                self.questionInfo, self.OpenAIChatBot
            )

            self.tokenCount = self.OpenAIChatBot.tokenUsage
            logging.info(
                f"Question Generation Token Count: {self.tokenCount}",
            )

    def loadQuestionData(self):
        """
        Loads the question data from a file.
        """
        loadedData = dataLoader(
            self.config, "questionData", f" - {self.config.generationModel}"
        )
        if loadedData is None:
            loadedData = [None] * 3
        if len(loadedData) != 3:
            logging.warning(
                "Loaded data for Question Data is incomplete/broken. Data will be regenerated and saved."
            )
            loadedData = [None] * 3
        (
            self.clusteredTopics,
            self.questionInfo,
            self.responseInfo,
        ) = loadedData

    def saveQuestionData(self):
        """
        Saves the question data to a file.
        """
        dataSaver(
            (
                self.clusteredTopics,
                self.questionInfo,
                self.responseInfo,
            ),
            self.config,
            "questionData",
            f" - {self.config.generationModel}",
        )

    def printQuestions(self):
        """
        Prints the generated questions, answers, correct answer, and reason for a given response.

        This method iterates over the responseInfo DataFrame and processes each response to generate
        questions and related information. It then logs the topic title, insert point, question,
        answers, correct answer, and reason using the logging module.

        Note:
        - The responseInfo DataFrame should have the following columns: 'Response Data', 'Topic Title',
          and 'End'.
        - The 'Response Data' column should contain the response data in JSON format.

        Returns:
        None
        """
        for index, row in self.responseInfo.iterrows():
            response = row["Response Data"]
            response = response.strip("` \n")
            if response.startswith("json"):
                response = response[4:]

            try:
                parsedResponse = json.loads(response)
            except json.JSONDecodeError as e:
                logging.warn(
                    f"Error decoding JSON: {e} for received response: {parsedResponse}"
                )
                logging.warn(f"Question for topic: {row['Topic Title']} not generated.")
                continue

            logging.info(f"Topic: {row['Topic Title']}")
            logging.info(f"Insertion Point: {row['End'].strftime('%H:%M:%S')}")
            question = f"Question: {parsedResponse['question'][:100]+'...'}"
            answers = "Answers: \n\t" + "\n\t".join(
                f"{i+1}. {item}" for i, item in enumerate(parsedResponse["answers"])
            )
            correct = f"Correct Answer: {parsedResponse['answers'].index(parsedResponse['correct'])+1}. {parsedResponse['correct']}"
            reason = f"Reason: {parsedResponse['reason'][:100]+'...'}"
            logging.info("\n".join([question, answers, correct, reason]))

    def printTokenCount(self):
        """
        Prints the token count.
        """
        logging.info(
            f"Question Generation Token Count: {self.tokenCount}",
        )

    def saveToFile(self):
        """
        Saves the question data to a file.

        This method creates a directory for the output if it doesn't exist,
        and then saves the question data to a file named "Questions.txt" in
        the output directory.

        Returns:
            None
        """
        saveFolder = os.path.join(outputFolder, self.config.videoToUse)
        try:
            if not os.path.exists(saveFolder):
                logging.info(f"Creating directory for Output: {saveFolder}")
                os.makedirs(saveFolder)
        except OSError:
            logging.warn(
                f"Creation of the directory {saveFolder} failed. Data output will not be saved"
            )
        questionSavePath = os.path.join(
            outputFolder,
            self.config.videoToUse,
            f"Questions - {self.config.generationModel}.txt",
        )
        logging.info(f"Saving Question Data to file: {questionSavePath}")
        try:
            with open(questionSavePath, "w") as file:
                writeBERTopicDataToFile(file, self.config.videoToUse, self.responseInfo)
            logging.info(f"Question Data saved to file: {questionSavePath}")
        except OSError:
            logging.warn(f"Failed to save question data to file: {questionSavePath}")
        except Exception as e:
            logging.warn(f"Failed to save question data to file: {questionSavePath}")
            logging.warn(f"Error: {e}")


# Using a manual overwrite option for debugging.
def retrieveBERTopicQuestions(
    config, topicModeller=None, videoData=None, overwrite=False
):
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
    questionData = BERTopicQuestionData(config)
    if not config.overwriteQuestionData and not overwrite:
        questionData.makeQuestionData(load=True)

        if questionData.responseInfo is not None:
            logging.info("Question Data loaded from saved files.")
            logging.info(f"Question Data Count: {len(questionData.responseInfo)}")
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
    logging.info(f"Question Data Count: {len(questionData.responseInfo)}")
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
        # As segments are linear, the start time is the timestamp of the first topic in the segment,
        # And the end time is the timestamp of the next topic in the segment.
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
    getAllTopics = topicModeller.topicModel.get_topics()
    topicList = {topicID: getAllTopics[topicID][0][0] for topicID in getAllTopics}

    # This is to handle duplicate topics that share the same name, but have different subtopics.
    cleanTopicList = modifyDuplicateTopics(topicList)

    clusteredTopics["Topic Title"] = clusteredTopics["Topic"].apply(
        lambda topic: cleanTopicList[topic]
    )

    if clusteredTopics.shape[0] == 0:
        logging.info(
            "No clustered topics found. This error should be unlikely and caused by a bug."
        )
        sys.exit("Topic clustering error. Exiting...")

    logging.info(f"Clustered Topics data shape: {clusteredTopics.shape}")
    logging.info(f"Clustered Topics head:\n {clusteredTopics.head(3)}")

    return clusteredTopics


def modifyDuplicateTopics(topicList):
    """
    Modifies a dictionary of topics to handle duplicate entries.

    Args:
        topicList (dict): A dictionary containing topics as keys.

    Returns:
        dict: The modified dictionary with duplicate topics modified.

    Example:
        >>> topicList = {'topic1': 'Python', 'topic2': 'Java', 'topic3': 'Python'}
        >>> modifyDuplicateTopics(topicList)
        {'topic1': 'Python #1', 'topic2': 'Java', 'topic3': 'Python #2'}
    """
    count = {}
    firstDuplicate = {}
    for key in list(topicList.keys()):
        if topicList[key] in count:
            if count[topicList[key]] == 1:
                topicList[firstDuplicate[topicList[key]]] = (
                    f"{topicList[firstDuplicate[topicList[key]]]} #1"
                )
            count[topicList[key]] += 1
            topicList[key] = f"{topicList[key]} #{count[topicList[key]]}"
        else:
            count[topicList[key]] = 1
            firstDuplicate[topicList[key]] = key
    return topicList


def getRelevantRegions(clusteredTopics, questionCount=-1, minFrequency=2):
    """
    Retrieves relevant text regions based on the given parameters.

    Args:
        clusteredTopics (DataFrame): A DataFrame containing clustered topics and their frequencies.
        questionCount (int, optional): The number of questions to generate. Defaults to -1.
        minFrequency (int, optional): The minimum frequency of a topic to be considered relevant. Defaults to 2.

    Returns:
        DataFrame: A DataFrame containing the relevant text regions.

    Raises:
        SystemExit: If no relevant text regions are found.

    """
    if questionCount == -1:
        # Retrieve one question per topic, by finding the text with the highest occurence of that topic.
        logging.info(f"No question count set. Retrieving one question per topic.")
        relevantRegions = []
        for group in clusteredTopics.groupby("Topic"):
            relevantRegions.append(
                group[1].sort_values("Frequency", ascending=False).head(1)
            )
        relevantRegions = pd.concat(relevantRegions).sort_values("Start")

    else:
        # Retrieve the relevant text regions sorted by frequency of topic occurence.
        logging.info(f"Identifying relevant text for {questionCount} questions.")
        sortedTopics = clusteredTopics.sort_values(
            ["Frequency", "Topic"], ascending=[False, True]
        )
        sortedTopics = sortedTopics[sortedTopics["Frequency"] >= minFrequency]

        if questionCount > sortedTopics.shape[0]:
            logging.info(
                f"""Question count is greater than the number of regions with relevant text. 
                {sortedTopics.shape[0]} questions will be generated instead."""
            )
        relevantRegions = sortedTopics.head(questionCount)

    if relevantRegions.shape[0] == 0:
        logging.error(
            "No relevant text regions found. This error should be unlikely and caused by a bug."
        )
        sys.exit("No relevant text regions found. Exiting...")

    logging.info(f"Relevant text regions data shape: {relevantRegions.shape}")
    logging.info(f"Relevant text regions head:\n {relevantRegions.head(3)}")

    return relevantRegions.reset_index(drop=True)


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


def truncateRelevantText(relevantRegions, videoData, contextWindowSize=600):
    """
    Truncates the relevant text regions based on the context window size.

    Args:
        relevantRegions (pandas.DataFrame): DataFrame containing the relevant text regions.
        videoData (pandas.DataFrame): DataFrame containing the video data.
        contextWindowSize (int, optional): Size of the context window in seconds. Defaults to 600.

    Returns:
        pandas.DataFrame: DataFrame containing the truncated relevant text regions.
    """
    truncatedRegions = []
    for index, region in relevantRegions.iterrows():
        relevantTextDuration = region["End"] - region["Start"]
        if contextWindowSize != 0 and relevantTextDuration > pd.Timedelta(
            seconds=contextWindowSize
        ):
            logging.info(
                f"Relevant text for Topic: {region['Topic Title']} is longer than context window size of {contextWindowSize} seconds. Truncating..."
            )
            # Find start of relevant truncated text
            truncatedSentences = videoData.combinedTranscript[
                (
                    videoData.combinedTranscript["End"]
                    >= (region["End"] - pd.Timedelta(seconds=contextWindowSize))
                )
            ]
            # Store the original start time of the topic
            region["Original Start"] = region["Start"]
            region["Start"] = truncatedSentences["Start"].min()

        truncatedRegions.append(region)
    truncatedRegions = pd.DataFrame(truncatedRegions)

    if truncatedRegions.shape[0] == 0:
        logging.error(
            "Truncated data has no data. This error should be unlikely and caused by a bug."
        )
        sys.exit("No truncated text found. Exiting...")

    return truncatedRegions


def getTextAndQuery(relevantRegions, videoData):
    """
    Retrieves relevant text from the combined transcript based on the given relevantRegions.

    Args:
        relevantRegions (DataFrame): The relevant regions containing the relevant topics.
        videoData (DataFrame): The combined transcript containing the relevant sentences.

    Returns:
        dict: A dictionary containing the relevant text for each region.
    """
    textAndQuery = []
    for index, region in relevantRegions.iterrows():
        transcriptSlice = videoData.combinedTranscript[
            (videoData.combinedTranscript["Start"] >= region["Start"])
            & (videoData.combinedTranscript["End"] <= region["End"])
        ]
        relevantSentences = " ".join(transcriptSlice["Combined Lines"].tolist())
        questionQuery = questionTaskBuilder(region["Topic Title"], relevantSentences)

        logging.info(
            f"Time range for relevant text for question - {index+1}: {region['Start'].strftime('%H:%M:%S')} to {region['End'].strftime('%H:%M:%S')}"
        )
        # logging.info(
        #     f"Relevant text selected for question - {index+1}: {relevantSentences}"
        # )
        logging.info(
            f"Query to generate question for Question {index+1} - First 100 Characters: {questionQuery[:100]}"
        )
        textAndQuery.append((relevantSentences, questionQuery))

    relevantRegions[["Relevant Text", "Question Query"]] = textAndQuery

    return relevantRegions


def getQuestionDataFromResponse(questionInfo, OpenAIChatBot):
    """
    Generates the question data using the OpenAI chatbot.

    This is because this is a much simpler task than the Topic Extraction part.
    Using LangChain in BERTopic lets BERTopic handle a lot of the complexities in how the topics are generated by leveraging LangChain.
    THat process involves passing keywords, relevant documents, and such to get the human-interpretable topics.

    In the case of Question generation, we only need to send one query per question we need to generate.
    We can get much finer control on the prompt directly because of the simpler query call made.

    LangChain is just a complicated way of building the prompt and storing information at its fundamental level as well.
    It just hides it from us through abstraction. This is a bit of an oversimplification, but it's a good way to think about it.
    So in this case, it made sense to stick to an OpenAIBot for our call rather than using LangChain.

    https://www.singlestore.com/blog/beginners-guide-to-langchain/
    """
    responseInfo = questionInfo.copy(deep=True)
    responseInfo["Response Data"] = None

    for index, row in questionInfo.iterrows():
        response = OpenAIChatBot.getResponse(row["Question Query"])
        responseInfo["Response Data"][index] = response[0]

    return responseInfo.reset_index(drop=True)


def writeBERTopicDataToFile(file, videoToUse, responseInfo):
    """
    Write information about response data to a file.

    Args:
        file (file object): The file object to write the information to.
        videoToUse (str): The name of the video or parent folder.
        responseInfo (pandas.DataFrame): The response data containing information about questions.

    Returns:
        None
    """
    file.write(f"Video Name / Parent Folder: {videoToUse}\n")
    for index, row in responseInfo.iterrows():
        response = row["Response Data"]
        response = response.strip("` \n")
        if response.startswith("json"):
            response = response[4:]

        startTime = row["Start"]
        endTime = row["End"]
        durationMin, durationSec = divmod((endTime - startTime).total_seconds(), 60)

        file.write("\n---------------------------------------\n")
        file.write(f"Question {index+1}\n")
        file.write(f"Topic: {row['Topic Title']}\n")

        # Retrieve the keywords for the topic.
        keywords = row["Words"]
        file.write(f"Keywords: {keywords}\n\n")

        file.write(
            f"Transcipt Segment: {startTime.strftime('%H:%M:%S')}"
            + f" - {endTime.strftime('%H:%M:%S')}\n"
        )
        file.write(
            f"Duration: {int(durationMin)} minutes & {int(durationSec)} seconds\n"
        )

        if "Original Start" in row and type(row["Original Start"]) == type(
            row["Start"]
        ):
            trueDurationMin, trueDurationSec = divmod(
                (endTime - row["Original Start"]).total_seconds(), 60
            )
            file.write(
                f"\tTruncated from original duration of {int(trueDurationMin)} minutes & {int(trueDurationSec)} seconds\n"
            )

        file.write(f"Insertion Point: {row['End'].strftime('%H:%M:%S')}\n")

        try:
            parsedResponse = json.loads(response)
            question = f"\nQuestion: {parsedResponse['question']}\n"
            answers = "Answers: \n\t" + "\n\t".join(
                [f"{i+1}. {item}" for i, item in enumerate(parsedResponse["answers"])]
            )
            correct = f"Correct Answer: \n\t{parsedResponse['answers'].index(parsedResponse['correct'])+1}. {parsedResponse['correct']}"
            reason = f"Reason: {parsedResponse['reason']}\n"
            file.write("\n".join([question, answers, correct, reason]))

        except json.JSONDecodeError as e:
            logging.warn(
                f"Error decoding JSON: {e} for received response: {parsedResponse}"
            )
            logging.warn(
                f"Question {index+1} for topic: {row['Topic Title']} not generated."
            )

            response = (
                f"Error decoding JSON: {e} for received response: {parsedResponse}\n"
                + f"Question {index+1} for topic: {row['Topic Title']} not generated.\n"
            )
            file.write(response)
