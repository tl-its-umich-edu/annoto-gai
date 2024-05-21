from config import Config
from transcriptLoader import TranscriptData
from topicExtractor import retrieveTopics
from utils import printAndLog


def main():
    config = Config()
    config.setFromEnv()

    printAndLog(f"Retrieving Transcript for {config.videoToUse}")

    videoData = TranscriptData(config)

    printAndLog(f"Retrieving Topics for {config.videoToUse}")

    topicModeller = retrieveTopics(config, videoData, overwrite=False)

if __name__ == "__main__":
    main()