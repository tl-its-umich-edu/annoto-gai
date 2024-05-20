from config import Config
from transcriptLoader import getCombinedTranscripts
from topicExtractor import retrieveTopics
from utils import printAndLog


def main():
    config = Config()
    config.setFromEnv()

    config.videoToUse = "New Google Assignments in Canvas"

    printAndLog(f"Retrieving Transcript for {config.videoToUse}")

    transcriptToUse = getCombinedTranscripts(
        config.videoToUse,
        captionsFolder=config.captionsFolder,
    )

    printAndLog(f"Retrieving Topics for {config.videoToUse}")

    topicsOverTime, topicModel = retrieveTopics(config.videoToUse, transcriptToUse, config=config)

if __name__ == "__main__":
    main()