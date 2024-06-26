from configData import configVars
from questionGenerator import processCaptions
import logging

logger = logging.getLogger(__name__)

def main():
    config = configVars()
    config.setFromEnv()

    processCaptions(config)


if __name__ == "__main__":
    main()
