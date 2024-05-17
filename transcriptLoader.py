import glob
from datetime import datetime
import pandas as pd


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
