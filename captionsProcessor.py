from configData import configVars
from transcriptLoader import retrieveTranscript
from topicExtractor import retrieveTopics
from questionGenerator import retrieveQuestions
import logging

logger = logging.getLogger(__name__)

def main():
    config = configVars()
    config.setFromEnv()

    logging.info(f"Retrieving Transcript for {config.videoToUse}...")
    videoData = TranscriptData(config)

    logging.info(f"Retrieving Topics for {config.videoToUse}...")
    topicModeller = retrieveTopics(config, videoData)
    topicModeller.printTopics()

    logging.info(f"Retrieving Questions for {config.videoToUse}...")
    questionData = retrieveQuestions(config, topicModeller, videoData)
    questionData.printQuestions()


if __name__ == "__main__":
    main()
