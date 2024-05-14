import logging
import os
import sys
from datetime import datetime
import glob
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from openai import AzureOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI as langchainAzureOpenAI


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
    start_time = ""
    end_time = ""

    for line in lines:
        line = line.strip()
        if line.isdigit():
            continue
        elif "-->" in line:
            start_time, end_time = line.split("-->")
            start_time = datetime.strptime(start_time.strip(), "%H:%M:%S,%f")  # .time()
            end_time = datetime.strptime(end_time.strip(), "%H:%M:%S,%f")  # .time()
        elif line:
            sentence += " " + line
        else:
            transcript.append(
                {"Line": sentence.strip(), "Start": start_time, "End": end_time}
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


def getcombinedTranscripts(
    captionsFolder="Captions",
    videoNames=["New Quizzes Video", "Rearrange Playlist video"],
):
    """
    Retrieves combined transcripts for multiple videos.

    Args:
        captionsFolder (str, optional): The folder where the caption files are located. Defaults to "Captions".
        videoNames (list, optional): The names of the videos for which to retrieve the transcripts. Defaults to ["New Quizzes Video", "Rearrange Playlist video"].

    Returns:
        dict: A dictionary containing the combined transcripts for each video.
    """
    srtFiles = {}
    transcripts = {}
    sentences = {}
    combinedTranscripts = {}
    for video in videoNames:
        srtFiles[video] = glob.glob(f"{captionsFolder}/{video}/*.srt")
        transcripts[video] = processSrtFile(srtFiles[video])
        sentences[video] = " ".join(transcripts[video]["Line"].tolist())
        combinedTranscripts[video] = lineCombiner(transcripts[video], windowSize=30)

    return combinedTranscripts


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


class Config:
    """
    A class that represents the configuration settings for the application.

    Attributes:
        logLevel (int): The logging level for the application.
        openAIParams (dict): A dictionary containing the OpenAI API parameters.
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

    def set(self, name, value):
        if name in self.__dict__:
            self.name = value
        else:
            raise NameError("Name not accepted in set() method")

    def configFetch(
        self, name, default=None, casting=None, validation=None, valErrorMsg=None
    ):
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
