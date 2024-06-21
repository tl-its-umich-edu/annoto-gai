import logging
import os
import sys
import time
from dotenv import load_dotenv

import openai
from openai import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    level=logging.INFO,
)

captionsFolder: str = "Captions"
saveFolder: str = "savedData"
savedFileTypes = ["transcriptData", "topicModel", "topicsOverTime", "questionData"]
outputFolder: str = "Output Data"
representationModelType: str = "langchain"

# Toggle for using KeyBERT vectorization in the BERTopic Model. Default is True.
useKeyBERT: bool = True

# Minimum threshold for the video duration in seconds for processing.
# Shorter videos might not have enough content to generate meaningful topics and questions.
# Default is 300s.
minVideoLength: int = 300

# Because of how BERTopic works, there needs to be a minimum number of sentences per window of time.
# When attempting to automatically segment the transcript into sentences using spaCy,
# If there are sentences longer than this duration, the transcript data might not be suitable for this process.
# Instead, the transcript will be segmented based on the `WINDOW_SIZE`.
# Defaults to 120s, must be >=60.
maxSentenceDuration = 120

# This sets the numbers of times a topic must appear in a given region of time to be considered a valid topic.
# This is used to filter out topics that are not relevant to the video content.
# Default is 2. Higher values will result in fewer questions possibly being generated.
minTopicFrequency: int = 2

for folder in savedFileTypes:
    folderPath = os.path.join(saveFolder, folder)
    try:
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
    except OSError:
        logging.error(f"Creation of the directory {folderPath} failed.")
        sys.exit(f"Directory creation failure. Exiting...")


class configVars:
    def __init__(self):
        self.logLevel = logging.INFO

        self.openAIParams: dict = {
            "KEY": "",
            "BASE": "",
            "VERSION": "",
            "MODEL": "",
            "ORGANIZATION": "",
        }

        self.videoToUse: str = ""
        self.questionCount: int = 3
        self.generationModel: str = "BERTopic"
        self.envImportSuccess: dict = {}

        # BERTopic-specific Parameters
        self.windowSize: int = 30
        self.contextWindowSize: int = 600

        self.overwriteTranscriptData: bool = False
        self.overwriteTopicModel: bool = False
        self.overwriteQuestionData: bool = False

        self.langchainPrompt: str = (
            "Give a single label that is only a few words long to summarize what these documents are about."
        )
        self.questionPrompt: str = (
            "You are a question-generating bot that generates questions for a given topic based on the provided relevant trancription text from a video."
        )

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
                if casting is bool:
                    value = int(value)
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

        This method reads configuration parameters from environment variables and sets them in the object.
        It also performs validation and error handling for the configuration parameters.

        Raises:
            ValueError: If there is a casting error or validation error for any configuration parameter.
            FileNotFoundError: If the .env file is missing.

        """
        if not os.path.exists(".env"):
            logging.error(
                "No .env file found. Please configure your environment variables use the .env.sample file as a template."
            )
            sys.exit("Missing .env file. Exiting...")

        # Force the environment variables to be read from the .env file every time.
        load_dotenv(".env", override=True)

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

        for credPart in self.openAIParams:
            if credPart == "BASE":
                envVarName = "AZURE_OPENAI_ENDPOINT"
            else:
                envVarName = "OPENAI_API_" + credPart

            self.openAIParams[credPart] = self.configFetch(
                envVarName,
                self.openAIParams[credPart],
                str,
                lambda param: len(param) > 0,
            )
            self.envImportSuccess[self.openAIParams[credPart]] = (
                False if not self.openAIParams[credPart] else True
            )

        if len(self.videoToUse) == 0:
            self.videoToUse = self.configFetch(
                "VIDEO_TO_USE",
                self.videoToUse,
                str,
                lambda name: len(name) > 0,
            )
            self.envImportSuccess[self.videoToUse] = (
                False if not self.videoToUse else True
            )

        self.questionCount = self.configFetch(
            "QUESTION_COUNT",
            self.questionCount,
            int,
            lambda x: x > 0 or x == -1,
        )
        self.envImportSuccess[self.questionCount] = (
            False if not self.questionCount else True
        )

        self.generationModel = self.configFetch(
            "GENERATION_MODEL",
            self.generationModel,
            str,
            lambda model: model in ["BERTopic", "LangChain"],
        )
        self.envImportSuccess[self.generationModel] = (
            False if not self.generationModel else True
        )

        self.overwriteTranscriptData = self.configFetch(
            "OVERWRITE_EXISTING_TRANSCRIPT",
            self.overwriteTranscriptData,
            bool,
            None,
        )
        self.envImportSuccess[self.overwriteTranscriptData] = (
            False if type(self.overwriteTranscriptData) is not bool else True
        )

        self.overwriteQuestionData = self.configFetch(
            "OVERWRITE_EXISTING_QUESTIONS",
            self.overwriteQuestionData,
            bool,
            None,
        )
        self.envImportSuccess[self.overwriteQuestionData] = (
            False if type(self.overwriteQuestionData) is not bool else True
        )

        if self.overwriteTranscriptData == True:
            self.overwriteQuestionData = True
            logging.info(
                "Generated Question data will also be overwritten as Transcript data is being overwritten."
            )

        # Pushed BERTopic variables out to a seperate method to keep the main method clean.
        if self.generationModel == "BERTopic":
            self.setBERTopicVarsFromEnv()

        if False in self.envImportSuccess.values():
            sys.exit("Configuration parameter import problems. Exiting...")

        logging.info("All configuration parameters set up successfully.")

    def setBERTopicVarsFromEnv(self):
        """
        Sets the BERTopic variables from the environment configuration.

        This method fetches the values of various BERTopic variables from the environment configuration.
        It validates the fetched values and updates the corresponding instance variables.
        It also updates the `envImportSuccess` dictionary to indicate whether the import was successful for each variable.

        Returns:
            None
        """
        self.windowSize = self.configFetch(
            "WINDOW_SIZE",
            self.windowSize,
            int,
            lambda x: x > 0,
        )
        self.envImportSuccess[self.windowSize] = False if not self.windowSize else True

        self.contextWindowSize = self.configFetch(
            "RELEVANT_TEXT_CONTEXT_WINDOW",
            self.contextWindowSize,
            int,
            lambda x: x >= 0,
        )
        self.envImportSuccess[self.contextWindowSize] = (
            False if self.contextWindowSize is None else True
        )

        self.overwriteTranscriptData = self.configFetch(
            "OVERWRITE_EXISTING_TRANSCRIPT",
            self.overwriteTranscriptData,
            bool,
            None,
        )
        self.envImportSuccess[self.overwriteTranscriptData] = (
            False if type(self.overwriteTranscriptData) is not bool else True
        )

        self.overwriteQuestionData = self.configFetch(
            "OVERWRITE_EXISTING_QUESTIONS",
            self.overwriteQuestionData,
            bool,
            None,
        )
        self.envImportSuccess[self.overwriteQuestionData] = (
            False if type(self.overwriteQuestionData) is not bool else True
        )

        self.langchainPrompt = self.configFetch(
            "LANGCHAIN_PROMPT",
            self.langchainPrompt,
            str,
            lambda prompt: len(prompt) > 0,
        )
        self.envImportSuccess[self.langchainPrompt] = (
            False if not self.langchainPrompt else True
        )

        self.questionPrompt = self.configFetch(
            "QUESTION_PROMPT",
            self.questionPrompt,
            str,
            lambda prompt: len(prompt) > 0,
        )
        self.envImportSuccess[self.questionPrompt] = (
            False if not self.questionPrompt else True
        )

        # This checks to set data in the later stages to be overwritten if the earlier stages are set to be overwritten.
        if self.overwriteTranscriptData == True:
            self.overwriteTopicModel = True
            logging.info(
                "Topic Model data will also be overwritten as Transcript data is being overwritten."
            )
        if self.overwriteTopicModel == True:
            self.overwriteQuestionData = True
            logging.info(
                "Generated Question data will also be overwritten as Topic Model data is being overwritten."
            )


class OpenAIBot:
    """
    A class representing an OpenAI chatbot.

    Attributes:
        config (object): The configuration object for the chatbot.
        messages (list): A list to store the chat messages.
        model (str): The OpenAI model to use for generating responses.
        systemPrompt (str): The system prompt to include in the chat messages.
        client (object): The AzureOpenAI client for making API calls.
        tokenUsage (int): The total number of tokens used by the chatbot.
        callMaxLimit (int): The maximum number of API call attempts allowed.

    Methods:
        getResponse(prompt): Generates a response for the given prompt.

    """

    def __init__(self, config):
        """
        Initializes a new instance of the OpenAIBot class.

        Args:
            config (object): The configuration object for the chatbot.

        """
        self.config = config
        self.messages = []
        self.model = self.config.openAIParams["MODEL"]
        self.systemPrompt = self.config.questionPrompt
        self.client = AzureOpenAI(
            api_key=self.config.openAIParams["KEY"],
            api_version=self.config.openAIParams["VERSION"],
            azure_endpoint=self.config.openAIParams["BASE"],
            organization=self.config.openAIParams["ORGANIZATION"],
        )
        self.tokenUsage = 0
        self.callMaxLimit = 3

    def getResponse(self, prompt):
        """
        Generates a response for the given prompt.

        Args:
            prompt (str): The user prompt for the chatbot.

        Returns:
            tuple: A tuple containing the response text and a boolean indicating if the response was successful.

        """
        callComplete = False
        callAttemptCount = 0
        while not callComplete and callAttemptCount < self.callMaxLimit:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.systemPrompt,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    stop=None,
                )
                time.sleep(1)
                callComplete = True

            except openai.AuthenticationError as e:
                logging.error(f"Error Message: {e}")
                sys.exit("Invalid OpenAI credentials. Exiting...")
            except openai.RateLimitError as e:
                logging.error(f"Error Message: {e}")
                logging.error("Rate limit hit. Pausing for a minute.")
                time.sleep(60)
                callComplete = False
            except openai.Timeout as e:
                logging.error(f"Error Message: {e}")
                logging.error("Timed out. Pausing for a minute.")
                time.sleep(60)
                callComplete = False
            except Exception as e:
                logging.error(f"Error Message: {e}")
                logging.error("Failed to send message. Trying again.")
                callComplete = False
            callAttemptCount += 1

        if callAttemptCount >= self.callMaxLimit:
            logging.error(
                f"Failed to send message at max limit of {self.callMaxLimit} times."
            )
            sys.exit("Too many failed attempts. Exiting...")

        elif callComplete:
            responseText = response.choices[0].message.content
            self.tokenUsage += response.usage.total_tokens

            return responseText, True


class LangChainBot:
    def __init__(self, config):
        """
        Initializes an instance of the LangChainBot class.

        Args:
            config (dict): A dictionary containing configuration parameters.

        Attributes:
            config (dict): The configuration parameters.
            model (str): The model specified in the configuration parameters.
            client (None): The client object (initially set to None).
            embeddings (None): Used only in LangChain-based Question Generation.
            chain (None): Used only in BERTopic-based Question Generation.
            tokenUsage (int): The token usage count.

        """
        self.config = config
        self.model = self.config.openAIParams["MODEL"]
        self.client = None

        self.embeddings = None  # Used only in LangChain-based Question Generation
        self.chain = None  # Used only in BERTopic-based Question Generation
        self.tokenUsage = 0

        self.initialize()

    def initialize(self):
        """
        Initializes the LangChainBot instance by calling the appropriate initialization methods based on the generation model specified in the configuration parameters.
        """
        self.initializeClient()

        if self.config.generationModel == "BERTopic":
            self.initializeChain()

        elif self.config.generationModel == "LangChain":
            self.initializeEmbeddings()

        else:
            logging.error(
                f"Invalid generation model specified: {self.config.generationModel}, valid options are 'BERTopic' and 'LangChain'."
            )
            sys.exit("Invalid generation model specified. Exiting...")

    def initializeClient(self):
        """
        Initializes the client object using the configuration parameters.
        """
        self.client = AzureChatOpenAI(
            api_key=self.config.openAIParams["KEY"],
            api_version=self.config.openAIParams["VERSION"],
            azure_endpoint=self.config.openAIParams["BASE"],
            organization=self.config.openAIParams["ORGANIZATION"],
            azure_deployment=self.config.openAIParams["MODEL"],
            temperature=0,
        )

    def initializeEmbeddings(self):
        """
        Initializes the embeddings object used for LangChain-based Question Generation using the configuration parameters.
        """
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=self.config.openAIParams["KEY"],
            api_version=self.config.openAIParams["VERSION"],
            azure_endpoint=self.config.openAIParams["BASE"],
            organization=self.config.openAIParams["ORGANIZATION"],
            azure_deployment="text-embedding-ada-002",  # This does not work if set to 'gpt-4', but seems to related to 'gpt-4' being the model used in the client.
        )

    def initializeChain(self):
        """
        Initializes the chain object used for BERTopic-based Question Generation using the client object.
        """
        self.chain = load_qa_chain(
            self.client,
            chain_type="stuff",
        )
