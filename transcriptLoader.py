import glob
import os
import sys
import logging
import pandas as pd
from datetime import datetime
from configData import captionsFolder, minVideoLength, maxSentenceDuration
from utils import dataLoader, dataSaver
import spacy


class TranscriptData:
    """
    Class to handle transcript data.

    Attributes:
        config (object): Configuration object.
        srtFiles (list): List of SRT files.
        transcript (object): Processed transcript data.
        processedSentences (object): Processed sentences.
        combinedTranscript (object): Combined transcript data.
    """

    def __init__(self, config):
        self.config = config
        self.srtFiles = None
        self.transcript = None
        self.processedSentences = None
        self.combinedTranscript = None

    def initialize(self, config):
        """
        Initializes the TranscriptData object with the given configuration.

        Args:
            config (object): Configuration object.
        """
        self.config = config

    def makeTranscriptData(self, load=True):
        """
        Generates the transcript data.

        Args:
            load (bool, optional): Whether to load existing transcript data. Defaults to True.
        """
        if load:
            self.loadTranscriptData()
        else:
            self.srtFiles = self.validateVideoFiles()
            self.transcript = processSrtFiles(self.srtFiles)
            self.processedSentences = getSentences(self.transcript)
            
            if self.processedSentences is not None:
                self.combinedTranscript = getCombinedTranscripts(
                    self.processedSentences, self.config.windowSize
                )
            else:
                # Revert to simple segmentation
                self.combinedTranscript = getCombinedTranscripts(
                    self.transcript, self.config.windowSize
                )
            self.saveTranscriptData()

    def loadTranscriptData(self):
        """
        Loads the transcript data from the data loader.
        """
        loadedData = dataLoader(self.config, "transcriptData")
        if loadedData is None:
            loadedData = [None] * 4
        elif type(loadedData) != tuple or len(loadedData) != 4:
            logging.warning(
                "Loaded data for Transcript Data is incomplete/broken. Data will be regenerated and saved."
            )
            loadedData = [None] * 4
        (
            self.srtFiles,
            self.transcript,
            self.processedSentences,
            self.combinedTranscript,
        ) = loadedData

    def saveTranscriptData(self):
        """
        Saves the transcript data using the data saver.
        """
        dataSaver(
            (
                self.srtFiles,
                self.transcript,
                self.processedSentences,
                self.combinedTranscript,
            ),
            self.config,
            "transcriptData",
        )

    def validateVideoFiles(self):
        """
        Validates the existence of video files and SRT files.

        Returns:
            list: List of validated SRT files.
        """
        if not os.path.exists(captionsFolder):
            os.makedirs(captionsFolder)
            logging.error(
                f"Captions folder not found. Created folder: {captionsFolder}."
            )
            sys.exit("Missing Captions parent folder. Exiting...")

        if not os.path.exists(os.path.join(captionsFolder, self.config.videoToUse)):
            logging.error(
                f"Video folder not found for {self.config.videoToUse} in Caption folder {captionsFolder}."
            )
            sys.exit("Missing Video folder. Exiting...")

        srtFiles = glob.glob(
            os.path.join(captionsFolder, self.config.videoToUse, "*.srt")
        )
        if len(srtFiles) == 0:
            logging.error(
                f"No SRT files found in {captionsFolder}/{self.config.videoToUse}."
            )
            sys.exit("No SRT files found. Exiting...")

        return srtFiles

    def printTranscript(self):
        """
        Prints the shape and head of the processed transcript data.
        """
        logging.info(
            f"Processed transcript data shape: {self.combinedTranscript.shape}"
        )
        logging.info(
            f"Processed transcript data head:\n {self.combinedTranscript.head(3)}"
        )


def processSrtFiles(srtFiles, minTranscriptLength=minVideoLength):
    """
    Process SRT files and extract transcript data.

    Args:
        srtFiles (list): A list of SRT file paths.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted transcript data.

    Raises:
        SystemExit: If no transcript data is found in the SRT file.

    """
    if len(srtFiles) > 1:
        logging.info(
            f"Multiple SRT files found. Using the first one: {srtFiles[0]}",
            LogOnly=True,
        )

    with open(srtFiles[0], "r") as f:
        lines = f.readlines()

    transcript = []

    timeFormat = "%H:%M:%S,%f"
    arrow = "-->"

    sentence = ""
    startTime, endTime = "", ""

    for line in lines:
        line = line.strip()
        if line.isdigit():
            continue
        elif arrow in line:
            startTime, endTime = line.split(arrow)
            startTime = datetime.strptime(startTime.strip(), timeFormat)  # .time()
            endTime = datetime.strptime(endTime.strip(), timeFormat)  # .time()
        elif line:
            sentence += " " + line
        else:
            transcript.append(
                {"Line": sentence.strip(), "Start": startTime, "End": endTime}
            )
            sentence = ""

    transcriptDF = pd.DataFrame(transcript)

    if transcriptDF.shape[0] == 0:
        logging.error(f"No transcript data found in {srtFiles[0]}. Exiting...")
        sys.exit("No transcript data found. Exiting...")

    if (transcriptDF["End"].iloc[-1] - transcriptDF["Start"].iloc[0]) < pd.Timedelta(
        seconds=minTranscriptLength
    ):
        logging.error(
            f"Video transcript is less than {minTranscriptLength} seconds long and not suitable for processing. Exiting..."
        )
        sys.exit(f"Transcript too short. Exiting...")

    logging.info(f"Transcript data extracted from {srtFiles[0]}")
    logging.info(f"Transcript data shape: {transcriptDF.shape}")
    logging.info(f"Transcript data head:\n {transcriptDF.head(3)}")

    return transcriptDF


def getSentences(transcript):
    """
    Extracts sentences from a transcript using the spaCy library.
    It contains a sentence boundary detection model that can attempt to parse and segment text into sentences.
    https://spacy.io/usage/linguistic-features#sbd

    Args:
        transcript (DataFrame): A pandas DataFrame containing the transcript data.

    Returns:
        DataFrame: A pandas DataFrame containing the extracted sentences with their start and end positions.
    """

    # Download the spaCy model using the following command:
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    parsedLines = nlp(" ".join(transcript["Line"].tolist()))

    startIndex, endIndex = 0, 1
    pastSentence = ""
    sentences = []
    for sentence in parsedLines.sents:
        sentenceMatched = False
        sentence = nlp(pastSentence + sentence.text)

        while not sentenceMatched and endIndex <= len(transcript):
            rows = transcript.iloc[startIndex:endIndex]

            if len(sentence.text) < len(" ".join(rows["Line"])):
                pastSentence = sentence.text + " "
                sentenceMatched = True
            elif sentence.text == " ".join(rows["Line"]):
                sentenceMatched = True
                sentences.append(
                    {
                        "Line": sentence.text,
                        "Start": transcript.iloc[startIndex]["Start"],
                        "End": transcript.iloc[endIndex - 1]["End"],
                    }
                )
                startIndex = endIndex
                pastSentence = ""
            else:
                endIndex += 1

    processedSentences = pd.DataFrame(sentences)

    if validateProcessedSentences(processedSentences, maxSentenceDuration):
        logging.info(f"Sentences data shape: {processedSentences.shape}")
        logging.info(f"Sentences data head:\n {processedSentences.head(3)}")

        return processedSentences
    else:
        return None


def validateProcessedSentences(
    processedSentences, maxSentenceDuration=maxSentenceDuration
):
    if processedSentences.shape[0] == 0:
        logging.warn(
            f"No sentences found in the transcript data. Reverting to simple segmentation.",
        )
        return False

    sentenceDurations = processedSentences["End"] - processedSentences["Start"]
    if sentenceDurations.max().seconds > maxSentenceDuration:
        logging.warn(
            f"Maximum sentence duration exceeds {maxSentenceDuration} seconds, indicating possibly bad transcription data. Reverting to simple segmentation.",
        )
        return False

    logging.info(f"Transcript successfully segmented into sentences.")
    return True


def getCombinedTranscripts(transcript, windowSize=30):
    """
    Combines overlapping transcripts within a given window size.

    Args:
        transcript (pandas.DataFrame): The input transcript data.
        windowSize (int, optional): The window size in seconds. Defaults to 30.

    Returns:
        pandas.DataFrame: The combined transcript data.

    """
    transcript = transcript.sort_values(by="Start")

    combinedTranscriptList = []
    currStart = transcript.iloc[0]["Start"]
    duration = pd.Timedelta(seconds=windowSize)

    while currStart < transcript.iloc[-1]["Start"]:
        slicedTranscript = transcript[
            (transcript["Start"] - currStart < duration)
            & (transcript["Start"] >= currStart)
        ]

        if slicedTranscript.shape[0] == 0:
            duration = pd.Timedelta(seconds=duration.seconds + 1)
            continue

        combinedLines = " ".join(slicedTranscript["Line"].tolist())
        combinedTranscriptList.append(
            {
                "Combined Lines": combinedLines,
                "Start": slicedTranscript.iloc[0]["Start"],
                "End": slicedTranscript.iloc[-1]["End"],
            }
        )

        currStart = slicedTranscript.iloc[-1]["End"]
        duration = pd.Timedelta(seconds=windowSize)

    combinedTranscript = pd.DataFrame(combinedTranscriptList)
    logging.info(
        f"Combined Transcript data shape: {combinedTranscript.shape}",
    )
    logging.info(f"Combined Transcript data head:\n {combinedTranscript.head(3)}")

    return combinedTranscript


def retrieveTranscript(config, overwrite=False):
    """
    Retrieves the transcript data based on the given configuration.

    Args:
        config (object): The configuration object containing the necessary parameters.
        overwrite (bool, optional): Flag indicating whether to overwrite existing transcript data. Defaults to False.
        Note that overwrite is used only for debugging purposes and should not be set to True in production.
        Use the OVERWRITE_EXISTING_TRANSCRIPT parameter to overwrite data.

    Returns:
        object: The TranscriptData object containing the retrieved transcript data.
    """

    transcriptData = TranscriptData(config)
    if not config.overwriteTranscriptData and not overwrite:
        transcriptData.makeTranscriptData(load=True)
        if transcriptData.combinedTranscript is not None:
            logging.info("Transcript Data loaded from saved files.")
            logging.info(
                f"Transcript Data head:\n {transcriptData.combinedTranscript.head(3)}"
            )
            return transcriptData

    logging.info("Generating & saving Transcript Data...")
    transcriptData.makeTranscriptData(load=False)

    return transcriptData
