import logging
import os
import glob
import sys
import pickle
from datetime import datetime
import pandas as pd

from openai import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI as langchainAzureOpenAI

from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, OpenAI, LangChain

from dotenv import load_dotenv

load_dotenv()

import numpy

numpy.seterr(divide="ignore")


def processSrtFile(srtFile):
    """
    Process an SRT file and extract the transcript data.

    Args:
        srtFile (str or list): The path to the SRT file or a list of paths.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted transcript data with columns 'Line', 'Start', and 'End'.
    """
    if type(srtFile) == list:
        srtFile = srtFile[0]

    with open(srtFile, "r") as f:
        lines = f.readlines()

    transcript = []

    sentence = ""
    startTime, endTime = "", ""

    for line in lines:
        line = line.strip()
        if line.isdigit():
            continue
        elif "-->" in line:
            startTime, endTime = line.split("-->")
            startTime = datetime.strptime(startTime.strip(), "%H:%M:%S,%f")  # .time()
            endTime = datetime.strptime(endTime.strip(), "%H:%M:%S,%f")  # .time()
        elif line:
            sentence += " " + line
        else:
            transcript.append(
                {"Line": sentence.strip(), "Start": startTime, "End": endTime}
            )
            sentence = ""

    return pd.DataFrame(transcript)


def lineCombiner(transcript, windowSize=20):
    """
    Combines consecutive lines in a transcript within a specified time window.

    Args:
        transcript (pandas.DataFrame): The transcript data containing columns "Start" and "Line".
        windowSize (int, optional): The time window size in seconds. Defaults to 20.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined lines with their start and end times.
    """

    transcript = transcript.sort_values(by="Start")

    combinedTranscript = []

    currStart = transcript.iloc[0]["Start"]

    while currStart < transcript.iloc[-1]["Start"]:
        slicedTranscript = transcript[
            (transcript["Start"] - currStart < pd.Timedelta(seconds=windowSize))
            & (transcript["Start"] >= currStart)
        ]
        combinedLines = " ".join(slicedTranscript["Line"].tolist())
        combinedTranscript.append(
            {
                "Combined Lines": combinedLines,
                "Start": slicedTranscript.iloc[0]["Start"],
                "End": slicedTranscript.iloc[-1]["End"],
            }
        )

        currStart = slicedTranscript.iloc[-1]["End"]

    return pd.DataFrame(combinedTranscript)


def getCombinedTranscripts(
    videoNames,
    captionsFolder="Captions",
    windowSize=30,
):
    """
    Retrieves and combines transcripts from SRT files for the specified videos.

    Args:
        captionsFolder (str, optional): The folder where the SRT files are located. Defaults to "Captions".
        videoNames (list, optional): The names of the videos to retrieve transcripts for. Defaults to ["New Quizzes Video", "Rearrange Playlist video"].
        windowSize (int, optional): The size of the window in seconds for combining lines. Defaults to 30.

    Returns:
        dict: A dictionary containing the combined transcripts for each video, where the keys are the video names and the values are the combined transcripts.
    """

    if type(videoNames) == str:
        videoNames = [videoNames]

    srtFiles, transcripts, sentences, combinedTranscripts = {}, {}, {}, {}
    for video in videoNames:
        srtFiles[video] = glob.glob(f"{captionsFolder}/{video}/*.srt")
        transcripts[video] = processSrtFile(srtFiles[video])
        sentences[video] = " ".join(transcripts[video]["Line"].tolist())
        combinedTranscripts[video] = lineCombiner(
            transcripts[video], windowSize=windowSize
        )

    if len(videoNames) == 1:
        return combinedTranscripts[videoNames[0]]
    
    return combinedTranscripts


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

    if (
        (topicModel := dataLoader(config, "topicModel", videoToUse, saveNameAppend))
        is not False
        and (
            topicsOverTime := dataLoader(
                config, "topicsOverTime", videoToUse, saveNameAppend
            )
        )
        is not False
        and not overwrite
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
    Returns a CountVectorizer model initialized with a vocabulary extracted from the given documents.

    Parameters:
    - docs (list): A list of documents to extract keywords from.

    Returns:
    - vectorizer_model (CountVectorizer): A CountVectorizer model initialized with the extracted vocabulary.
    """
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
        representation_model = KeyBERTInspired()

    if config.representationModelType == "openai":
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
        LangChainQABot = LangChainBot(config)
        chain = LangChainQABot.chain
        prompt = "Give a single label that is only a few words long to summarize what these documents are about"
        representation_model = LangChain(chain, prompt=prompt)

    if config.useKeyBERT:
        topicModel = BERTopic(
            representation_model=representation_model,
            vectorizer_model=getVectorizer(docs),
        )
    else:
        topicModel = BERTopic(representation_model=representation_model)

    topics, probs = topicModel.fit_transform(docs)

    # hierarchical_topics = topic_model.hierarchical_topics(docs)
    # topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

    binCount = getBinCount(combinedTranscript, windowSize=120)
    topicsOverTime = topicModel.topics_over_time(docs, timestamps, nr_bins=binCount)

    # topicModel.visualize_topics_over_time(topicsOverTime)

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


def getMaxTopics(topicsOverTime, combinedTranscript):
    """
    Get the maximum topics based on frequency over time.

    Args:
        topicsOverTime (DataFrame): A DataFrame containing information about topics over time.

    Returns:
        dict: A dictionary containing the maximum topics with their corresponding details.

    """
    if "Name" not in topicsOverTime.columns:
        topicsOverTime["Name"] = topicsOverTime["Words"].apply(
            lambda wordList: f"Keywords: {wordList}"
        )

    timestampStack = []
    for group in topicsOverTime.groupby("Timestamp"):
        timestampStack.append(
            group[1].sort_values("Frequency", ascending=False).head(1)
        )

    binnedTopics = pd.concat(timestampStack).sort_values("Timestamp")
    binnedTopics = binnedTopics[binnedTopics["Topic"] != -1]

    binnedTopics = (
        binnedTopics.groupby(
            (binnedTopics["Topic"] != binnedTopics["Topic"].shift()).cumsum()
        )
        .agg(
            {
                "Topic": "first",
                "Words": "first",
                "Frequency": "sum",
                "Timestamp": "first",
                "Name": "first",
            }
        )
        .reset_index(drop=True)
    )

    binnedTopics["Start"] = binnedTopics["Timestamp"]
    binnedTopics["End"] = binnedTopics["Timestamp"].shift(-1)

    binnedTopics.at[binnedTopics.index[0], "Start"] = combinedTranscript["Start"].iloc[
        0
    ]
    binnedTopics.at[binnedTopics.index[-1], "End"] = combinedTranscript["End"].iloc[-1]

    binnedTopics = binnedTopics.drop("Timestamp", axis=1)

    maxStack = []
    for group in binnedTopics.groupby("Topic"):
        maxStack.append(group[1].sort_values("Frequency", ascending=False).head(1))

    maxTopics = pd.concat(maxStack).sort_values("Start")
    maxTopics["Cleaned Name"] = maxTopics["Name"].apply(
        lambda name: name.lstrip("0123456789_., ").rstrip("0123456789_., ")
    )
    maxTopics = maxTopics.set_index("Cleaned Name").to_dict(orient="index")

    return maxTopics


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


def getRelevantText(maxTopics, combinedTranscript):
    """
    Retrieves relevant text from the combined transcript based on the given maxTopics.

    Args:
        combinedTranscript (DataFrame): The combined transcript containing the relevant sentences.
        maxTopics (dict): A dictionary containing the maxTopics and their corresponding start and end times.

    Returns:
        dict: A modified version of the maxTopics dictionary with the relevant text and question text added.
    """
    for topic in maxTopics:
        relevantSentences = combinedTranscript[
            (combinedTranscript["Start"] > maxTopics[topic]["Start"])
            & (combinedTranscript["End"] < maxTopics[topic]["End"])
        ]
        maxTopics[topic]["Relevant Text"] = " ".join(
            relevantSentences["Combined Lines"].tolist()
        )
        maxTopics[topic]["Question Text"] = questionTaskBuilder(
            topic, maxTopics[topic]["Relevant Text"]
        )

    return maxTopics


def retrieveQuestionData(
    videoToUse, topicTexts, config, overwrite=False, saveNameAppend=""
):
    """
    Retrieve question data for a given video and topic texts.

    Args:
        videoToUse (str): The video to use for generating question data.
        topicTexts (list): List of topic texts to generate questions from.
        config (dict): Configuration settings for question generation.
        overwrite (bool, optional): Whether to overwrite existing question data. Defaults to False.
        saveNameAppend (str, optional): Additional string to append to the save name. Defaults to "".

    Returns:
        list: The generated question data.

    """
    if (
        questionData := dataLoader(config, "questionData", videoToUse, saveNameAppend)
    ) is not False and not overwrite:
        logging.info("Question Data loaded from saved files.")
        return questionData

    logging.info("Generating & saving Question Data.")
    questionData = questionGenerator(topicTexts, config)
    dataSaver(questionData, config, "questionData", videoToUse, saveNameAppend)

    return questionData


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
    if dataType in ["topicsOverTime", "questionData"]:
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
        if dataType in ["topicsOverTime", "questionData"]:
            return pickle.load(open(savePath, "rb"))

    return False


def questionGenerator(topicTexts, config):
    """
    Generates questions for a given topic based on the provided relevant transcription text from a video.

    Args:
        topicTexts (dict): A dictionary containing topic texts and corresponding question text.

    Returns:
        dict: A dictionary containing the updated topic texts with generated questions.
    """
    OpenAIChatBot = OpenAIBot(config)

    for topic in topicTexts:
        response = OpenAIChatBot.client.chat.completions.create(
            model=OpenAIChatBot.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a question-generating bot that generates questions for a given topic based on the provided relevant trancription text from a video.",
                },
                {"role": "user", "content": topicTexts[topic]["Question Text"]},
            ],
            temperature=0,
            stop=None,
        )

        responseText = response.choices[0].message.content

        topicTexts[topic]["JSON"] = responseText

    return topicTexts


class Config:
    """
    Configuration class for managing application settings.
    """

    def __init__(self):
        self.logLevel = logging.INFO

        self.openAIParams: dict = {
            "KEY": None,
            "BASE": None,
            "VERSION": None,
            "MODEL": None,
            "ORGANIZATION": None,
        }

        self.saveFolder = "savedData"
        self.fileTypes = ["topicModel", "topicsOverTime", "questionData"]

        for folder in self.fileTypes:
            folderPath = os.path.join(self.saveFolder, folder)
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)

        self.representationModelType = "langchain"
        self.useKeyBERT = True

    def set(self, name, value):
        """
        Set the value of a configuration parameter.

        Args:
            name (str): The name of the configuration parameter.
            value: The value to be set.

        Raises:
            NameError: If the name is not accepted in the `set()` method.
        """
        if name in self.__dict__:
            self.name = value
        else:
            raise NameError("Name not accepted in set() method")

    def configFetch(
        self, name, default=None, casting=None, validation=None, valErrorMsg=None
    ):
        """
        Fetch a configuration parameter from the environment variables.

        Args:
            name (str): The name of the configuration parameter.
            default: The default value to be used if the parameter is not found in the environment variables.
            casting (type): The type to cast the parameter value to.
            validation (callable): A function to validate the parameter value.
            valErrorMsg (str): The error message to be logged if the validation fails.

        Returns:
            The value of the configuration parameter, or None if it is not found or fails validation.
        """
        value = os.environ.get(name, default)
        if casting is not None:
            try:
                value = casting(value)
            except ValueError:
                errorMsg = f'Casting error for config item "{name}" value "{value}".'
                logging.error(errorMsg)
                return None

        if validation is not None and not validation(value):
            errorMsg = f'Validation error for config item "{name}" value "{value}".'
            logging.error(errorMsg)
            return None
        return value

    def setFromEnv(self):
        """
        Set configuration parameters from environment variables.
        """
        try:
            self.logLevel = str(os.environ.get("LOG_LEVEL", self.logLevel)).upper()
        except ValueError:
            warnMsg = f"Casting error for config item LOG_LEVEL value. Defaulting to {logging.getLevelName(logging.root.level)}."
            logging.warning(warnMsg)

        try:
            logging.getLogger().setLevel(logging.getLevelName(self.logLevel))
        except ValueError:
            warnMsg = f"Validation error for config item LOG_LEVEL value. Defaulting to {logging.getLevelName(logging.root.level)}."
            logging.warning(warnMsg)

        # Currently the code will check and validate all config variables before stopping.
        # Reduces the number of runs needed to validate the config variables.
        envImportSuccess = True

        for credPart in self.openAIParams:

            if credPart == "BASE":
                self.openAIParams[credPart] = self.configFetch(
                    "AZURE_OPENAI_ENDPOINT", self.openAIParams[credPart], str
                )
            else:
                self.openAIParams[credPart] = self.configFetch(
                    "OPENAI_API_" + credPart, self.openAIParams[credPart], str
                )
            envImportSuccess = (
                False
                if not self.openAIParams[credPart] or not envImportSuccess
                else True
            )

        if not envImportSuccess:
            sys.exit("Exiting due to configuration parameter import problems.")
        else:
            logging.info("All configuration parameters set up successfully.")


class OpenAIBot:
    """
    A class representing an OpenAI Bot.

    Attributes:
        config (dict): The configuration parameters for the bot.
        messages (list): A list to store the messages.
        model (str): The model used by the bot.
        client (AzureOpenAI): An instance of the AzureOpenAI client.

    Methods:
        __init__(self, config): Initializes a new instance of the OpenAIBot class.
    """

    def __init__(self, config):
        self.config = config
        self.messages = []

        self.model = self.config.openAIParams["MODEL"]

        self.client = AzureOpenAI(
            api_key=self.config.openAIParams["KEY"],
            api_version=self.config.openAIParams["VERSION"],
            azure_endpoint=self.config.openAIParams["BASE"],
            organization=self.config.openAIParams["ORGANIZATION"],
        )


class LangChainBot:
    """
    A class representing a language chain bot.

    Attributes:
        config (object): The configuration object for the bot.
        messages (list): A list to store messages.
        model (str): The model used by the bot.
        client (object): The client object for the Azure OpenAI service.
        chain (object): The QA chain object.

    Methods:
        __init__(self, config): Initializes the LangChainBot object.
    """

    def __init__(self, config):
        self.config = config
        self.messages = []

        self.model = self.config.openAIParams["MODEL"]

        self.client = langchainAzureOpenAI(
            api_key=self.config.openAIParams["KEY"],
            api_version=self.config.openAIParams["VERSION"],
            azure_endpoint=self.config.openAIParams["BASE"],
            organization=self.config.openAIParams["ORGANIZATION"],
            azure_deployment=self.config.openAIParams["MODEL"],
            temperature=0,
        )

        self.chain = load_qa_chain(
            self.client,
            chain_type="stuff",
        )
