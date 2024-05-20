import logging
import os
import sys
from dotenv import load_dotenv

from openai import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI as langchainAzureOpenAI

# from langchain_community.callbacks import get_openai_callback


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

            self.videoToUse = ''

        self.saveFolder = "savedData"
        self.fileTypes = ["topicModel", "topicsOverTime"]

        for folder in self.fileTypes:
            folderPath = os.path.join(self.saveFolder, folder)
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)

        self.representationModelType = "langchain"
        self.useKeyBERT = True

        self.topicTokenCount = 0
        self.questionTokenCount = 0

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

        self.videoToUse = self.configFetch(
            "VIDEO_TO_USE",
            None,
            str,
            lambda param: len(param) > 0,
        )
        envImportSuccess = False if not self.videoToUse or not envImportSuccess else True

        if not envImportSuccess:
            sys.exit("Exiting due to configuration parameter import problems.")
        else:
            logging.info("All configuration parameters set up successfully.")


class OpenAIBot:
    """
    A class representing an OpenAI chatbot.

    Attributes:
        config (object): The configuration object for the chatbot.
        messages (list): A list of messages exchanged between the user and the chatbot.
        model (str): The model used by the chatbot.
        systemPrompt (str): The system prompt used by the chatbot.
        client (object): The AzureOpenAI client used for making API calls.
        tokenUsage (int): The total number of tokens used by the chatbot.

    Methods:
        getResponse(prompt): Generates a response based on the given prompt using the chat model.

    """

    def __init__(self, config):
        self.config = config
        self.messages = []

        self.model = self.config.openAIParams["MODEL"]

        self.systemPrompt = "You are a question-generating bot that generates questions for a given topic based on the provided relevant trancription text from a video."

        self.client = AzureOpenAI(
            api_key=self.config.openAIParams["KEY"],
            api_version=self.config.openAIParams["VERSION"],
            azure_endpoint=self.config.openAIParams["BASE"],
            organization=self.config.openAIParams["ORGANIZATION"],
        )

        self.tokenUsage = 0

    def getResponse(self, prompt):
        """
        Generates a response based on the given prompt using the chat model.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            str: The generated response.

        """
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

        responseText = response.choices[0].message.content
        self.tokenUsage += response.usage.total_tokens

        return responseText


class LangChainBot:
    """
    A class representing a language chain bot.

    Attributes:
        config (object): The configuration object for the bot.
        messages (list): A list to store messages.
        model (str): The model used by the bot.
        client (object): The client object for the Azure OpenAI service.
        chain (object): The QA chain object.

    Methods:
        __init__(self, config): Initializes the LangChainBot object.
    """

    def __init__(self, config):
        self.config = config

        self.model = self.config.openAIParams["MODEL"]

        self.client = langchainAzureOpenAI(
            api_key=self.config.openAIParams["KEY"],
            api_version=self.config.openAIParams["VERSION"],
            azure_endpoint=self.config.openAIParams["BASE"],
            organization=self.config.openAIParams["ORGANIZATION"],
            azure_deployment=self.config.openAIParams["MODEL"],
            temperature=0,
        )

        self.chain = load_qa_chain(
            self.client,
            chain_type="stuff",
        )

        self.prompt = "Give a single label that is only a few words long to summarize what these documents are about"

        self.tokenUsage = 0
