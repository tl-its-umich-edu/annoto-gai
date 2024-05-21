import glob
import os
from datetime import datetime
import pandas as pd
from utils import printAndLog
import sys


class TranscriptData:
    """
    Class to handle transcript data.

    Args:
        config (object): Configuration object.

    Attributes:
        config (object): Configuration object.
        videoToUse (str): Name of the video to use.
        srtFiles (list): List of validated SRT files.
        transcript (DataFrame): Processed transcript data.
        combinedTranscript (DataFrame): Combined transcript data.

    Methods:
        validateVideoFiles: Validates the video files.
        processSrtFiles: Processes the SRT files and extracts transcript data.
        getCombinedTranscripts: Combines the transcript data based on a window size.
    """

    def __init__(self, config):
        self.config = config
        self.videoToUse = self.config.videoToUse

        self.srtFiles = self.validateVideoFiles()
        self.transcript = self.processSrtFiles()
        self.combinedTranscript = self.getCombinedTranscripts()

    def validateVideoFiles(self):
        """
        Validates the video files.

        Returns:
            list: List of validated SRT files.

        Raises:
            SystemExit: If video folder or SRT files are not found.
        """
        if not os.path.exists(
            os.path.join(self.config.captionsFolder, self.videoToUse)
        ):
            printAndLog(
                f"Video folder not found for {self.config.videoToUse} in Caption folder {self.config.captionsFolder}.",
                level="error",
            )
            sys.exit("Captions folder not found. Exiting...")

        srtFiles = glob.glob(
            os.path.join(self.config.captionsFolder, self.config.videoToUse, "*.srt")
        )
        if len(srtFiles) == 0:
            printAndLog(
                f"No SRT files found in {self.config.captionsFolder}/{self.config.videoToUse}.",
                level="error",
            )
            sys.exit("No SRT files found. Exiting...")

        return srtFiles

    def processSrtFiles(self):
        """
        Processes the SRT files and extracts transcript data.

        Returns:
            DataFrame: Processed transcript data.
        """
        if len(self.srtFiles) > 1:
            printAndLog(
                f"Multiple SRT files found. Using the first one: {self.srtFiles[0]}",
                LogOnly=True,
            )

        with open(self.srtFiles[0], "r") as f:
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
        printAndLog(f"Transcript data extracted from {self.srtFiles[0]}", logOnly=True)
        printAndLog(f"Transcript data shape: {transcriptDF.shape}", logOnly=True)
        printAndLog(f"Transcript data head: {transcriptDF.head(5)}", logOnly=True)

        if transcriptDF.shape[0] == 0:
            printAndLog(
                f"No transcript data found in {self.srtFiles[0]}. Exiting...",
                level="error",
            )
            sys.exit("No transcript data found. Exiting...")

        return transcriptDF

    def getCombinedTranscripts(self):
        """
        Combines the transcript data based on a window size.

        Returns:
            DataFrame: Combined transcript data.
        """
        transcript = self.transcript.sort_values(by="Start")

        combinedTranscript = []

        currStart = transcript.iloc[0]["Start"]
        duration = pd.Timedelta(seconds=self.config.windowSize)

        while currStart < transcript.iloc[-1]["Start"]:
            slicedTranscript = transcript[
                (transcript["Start"] - currStart < duration)
                & (transcript["Start"] >= currStart)
            ]

            if slicedTranscript.shape[0] == 0:
                duration = pd.Timedelta(seconds=duration.seconds + 1)
                continue

            combinedLines = " ".join(slicedTranscript["Line"].tolist())
            combinedTranscript.append(
                {
                    "Combined Lines": combinedLines,
                    "Start": slicedTranscript.iloc[0]["Start"],
                    "End": slicedTranscript.iloc[-1]["End"],
                }
            )

            currStart = slicedTranscript.iloc[-1]["End"]
            duration = pd.Timedelta(seconds=self.config.windowSize)

        combinedTranscript = pd.DataFrame(combinedTranscript)
        printAndLog(
            f"Combined Transcript data shape: {combinedTranscript.shape}", logOnly=True
        )
        printAndLog(
            f"Combined Transcript data head: {combinedTranscript.head(5)}", logOnly=True
        )

        return combinedTranscript
