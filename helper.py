import logging
import os
import glob
import sys
from datetime import datetime
import pandas as pd

from dotenv import load_dotenv


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
    Retrieves and combines transcripts from multiple videos.

    Args:
        videoNames (str or list): The name(s) of the video(s) to retrieve transcripts from.
        captionsFolder (str, optional): The folder where the caption files are located. Defaults to "Captions".
        windowSize (int, optional): The size of the window for combining lines in the transcripts. Defaults to 30.

    Returns:
        dict or str: If only one video name is provided, returns the combined transcript as a string.
                    If multiple video names are provided, returns a dictionary of combined transcripts for each video.
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


class Config:
    """
    Configuration class for managing application settings.
    """

    def __init__(self):
        self.logLevel = logging.INFO

        self.openAIParams: dict = {
            "KEY": "",
            "BASE": "",
            "VERSION": "",
            "MODEL": "",
            "ORGANIZATION": "",
        }

        self.captionsFolder = "Captions"
        if not os.path.exists(self.captionsFolder):
            os.makedirs(self.captionsFolder)

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

        if not os.path.exists(".env"):
            errorMsg = "No .env file found. Please configure your environment variables use the .env.sample file as a template."
            logging.error(errorMsg)
            sys.exit(errorMsg)

        load_dotenv()

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
                    "AZURE_OPENAI_ENDPOINT",
                    self.openAIParams[credPart],
                    str,
                    lambda param: len(param) > 0,
                )
            else:
                self.openAIParams[credPart] = self.configFetch(
                    "OPENAI_API_" + credPart,
                    self.openAIParams[credPart],
                    str,
                    lambda param: len(param) > 0,
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
