from config import Config
from transcriptLoader import getCombinedTranscripts
from topicExtractor import retrieveTopics
from utils import printAndLog


def main():
    config = Config()
    config.setFromEnv()

    videoToUse = "New Google Assignments in Canvas"

    printAndLog(f"Retrieving Transcript for {videoToUse}")

    transcriptToUse = getCombinedTranscripts(
        videoToUse,
        captionsFolder=config.captionsFolder,
    )

    printAndLog(f"Retrieving Topics for {videoToUse}")

    topicsOverTime, topicModel = retrieveTopics(videoToUse, transcriptToUse, config=config)

if __name__ == "__main__":
    main()