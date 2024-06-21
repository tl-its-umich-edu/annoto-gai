import sys
import os
import logging
from configData import outputFolder, LangChainBot
from utils import getMetadata, formatDocs, dataLoader, dataSaver

from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader


class Question(BaseModel):
    "Question Details"

    question: str = Field(title="Question", description="Question formed by the model.")
    answers: List[str] = Field(
        title="Answers",
        description="List of 4 possible answers to the question formed .",
    )
    correctAnswerIndex: int = Field(
        title="Correct Answer Index",
        description="Index of the correct answer to the question formed .",
    )
    reason: str = Field(
        title="Reason",
        description="Explanation for the correct answer to the question formed.",
    )
    topic: str = Field(title="Topic", description="Topic of the question formed.")
    insertionTime: str = Field(
        title="Insertion Time",
        description="Timestamp at which the question is to be inserted within a transcript, which is at the end of a relevant transcript section.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which was used to form the question.",
    )


class Questions(BaseModel):
    "List of formed Questions"

    questions: List[Question] = Field(
        title="Questions",
        description="List of questions formed for a given transcript.",
    )

    def print(self):
        for i, question in enumerate(self.questions):
            print(f"Question {i+1}: {question.question}")
            print(f"Answers: {question.answers}")
            print(
                f"Correct Answer: {question.correctAnswerIndex}: {question.answers[question.correctAnswerIndex]}"
            )
            print(f"Reason: {question.reason}")
            print(f"Topic: {question.topic}")
            print(f"Insertion Time: {question.insertionTime}")
            print(f"Citations: {question.citations}")
            print("\n")


class LangChainQuestionData:
    """
    Represents a class that handles question data generation for the LangChainBot.

    Args:
        config: The configuration object for the question generation.
        videoData: Optional video data object.

    Attributes:
        config: The configuration object for the question generation.
        videoData: The video data object.
        LangChainQuestionBot: The LangChainBot instance.
        retriever: The retriever object for question generation.
        runnable: The runnable object for question generation.
        responseInfo: The Questions object containing the generated question data.

    Methods:
        initialize: Initializes the LangChainQuestionData object.
        makeQuestionData: Generates the question data.
        loadQuestionData: Loads the question data from a file.
        saveQuestionData: Saves the question data to a file.
        saveToFile: Saves the question data to a file.

    """

    def __init__(self, config, videoData=None):
        self.config = config
        self.videoData = videoData
        self.LangChainQuestionBot = None
        self.retriever = None
        self.runnable = None
        self.responseInfo: Questions = None

    def initialize(self, videoData):
        """
        Initializes the LangChainQuestionData object.

        Args:
            videoData: The video data object.

        """
        self.videoData = videoData
        self.LangChainQuestionBot = LangChainBot(self.config)
        self.retriever = makeRetriever(
            self.videoData.combinedTranscript, self.LangChainQuestionBot.embeddings
        )
        self.runnable = makeRunnable(self.retriever, self.LangChainQuestionBot.client)

    def makeQuestionData(self, load=True):
        """
        Generates the question data.

        Args:
            load: Whether to load existing question data or generate new data.

        """
        if load:
            self.loadQuestionData()
        else:
            self.responseInfo = self.runnable.invoke(f"{self.config.questionCount}")

    def loadQuestionData(self):
        """
        Loads the question data from a file.
        """
        loadedData = dataLoader(
            self.config, "questionData", f" - {self.config.generationModel}"
        )
        if loadedData is None:
            loadedData = [None] * 1
        if len(loadedData) != 1:
            logging.warning(
                "Loaded data for Question Data is incomplete/broken. Data will be regenerated and saved."
            )
            loadedData = [None] * 1
        (self.responseInfo,) = loadedData

    def saveQuestionData(self):
        """
        Saves the question data to a file.
        """
        dataSaver(
            (self.responseInfo,),
            self.config,
            "questionData",
            f" - {self.config.generationModel}",
        )

    def printQuestions(self):
        """
        Prints the question data.
        """
        for index, question in enumerate(self.responseInfo.questions):
            logging.info("\n---------------------------------------\n")
            logging.info(f"Question {index+1}")
            logging.info(f"Topic: {question.topic}")
            logging.info(f"Insertion Point: {question.insertionTime}")
            logging.info(f"Question {index+1}: {question.question[:100]+'...'}")
            answers = "Answers: \n\t" + "\n\t".join(
                [f"{i+1}. {item}" for i, item in enumerate(question.answers)]
            )
            logging.info(f"{answers}")
            logging.info(
                f"Correct Answer: {question.correctAnswerIndex+1}. {question.answers[question.correctAnswerIndex]}"
            )
            logging.info(f"Reason: {question.reason[:100]+'...'}")
            logging.info(f"Citations: {question.citations}\n")

    def saveToFile(self):
        """
        Saves the question data to a file.
        """
        saveFolder = os.path.join(outputFolder, self.config.videoToUse)
        try:
            if not os.path.exists(saveFolder):
                logging.info(f"Creating directory for Output: {saveFolder}")
                os.makedirs(saveFolder)
        except OSError:
            logging.warn(
                f"Creation of the directory {saveFolder} failed. Data output will not be saved"
            )
        questionSavePath = os.path.join(
            outputFolder,
            self.config.videoToUse,
            f"Questions - {self.config.generationModel}.txt",
        )
        logging.info(f"Saving Question Data to file: {questionSavePath}")
        try:
            with open(questionSavePath, "w") as file:
                writeLangChainDataToFile(
                    file, self.config.videoToUse, self.responseInfo
                )
            logging.info(f"Question Data saved to file: {questionSavePath}")
        except OSError:
            logging.warn(f"Failed to save question data to file: {questionSavePath}")
        except Exception as e:
            logging.warn(f"Failed to save question data to file: {questionSavePath}")
            logging.warn(f"Error: {e}")


def retrieveLangChainQuestions(config, videoData=None, overwrite=False):

    questionData = LangChainQuestionData(config)
    if not config.overwriteQuestionData and not overwrite:
        questionData.makeQuestionData(load=True)

        if questionData.responseInfo is not None:
            logging.info("Question Data loaded from saved files.")
            logging.info(f"Question Data Count: {len(questionData.responseInfo.questions)}")
            return questionData

    if videoData is None:
        logging.error(
            "No saved data was found, and no video data was provided in function call needed to extract topics."
        )
        sys.exit("Video Data not provided. Exiting...")
    else:
        questionData.initialize(videoData)

    logging.info("Generating Question Data...")
    questionData.makeQuestionData(load=False)
    questionData.saveQuestionData()

    logging.info("Question Data generated and saved for current configuration.")
    logging.info(f"Question Data Count: {len(questionData.responseInfo.questions)}")
    return questionData


def makeRetriever(transcript, embeddings) -> Chroma:
    transcript = getMetadata(transcript)
    loader = DataFrameLoader(transcript, page_content_column="Combined Lines")
    vectorstore = Chroma.from_documents(documents=loader.load(), embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def makeRunnable(retriever, client):
    template = """You are a question-generating algorithm.
                Only extract relevant information from the provided trancription text: {context}
                Generate {count} Multiple-Choice Questions with 4 possible answers for each question, and provide a reason for the correct answer.
                Provide an appropriate timestamp to show where each question would be inserted within the transcript.
                This is at the end of the relevant text section used to form the question, using the metadata information.
                Try to cover a wide range of topics covered in the tranacription text.
                The questions should be in line with the overall theme of the text."""
    prompt = ChatPromptTemplate.from_template(template)
    runnable = (
        {"context": retriever | formatDocs, "count": RunnablePassthrough()}
        | prompt
        | client.with_structured_output(schema=Questions)
    )
    return runnable


def writeLangChainDataToFile(file, videoName, responseInfo):
    file.write(f"Video Name / Parent Folder: {videoName}\n")
    for index, question in enumerate(responseInfo.questions):
        file.write("\n---------------------------------------\n")
        file.write(f"Question {index+1}\n")
        file.write(f"Topic: {question.topic}\n")

        file.write(f"Insertion Point: {question.insertionTime}\n")

        file.write(f"Question {index+1}: {question.question}\n")

        answers = "Answers: \n\t" + "\n\t".join(
            [f"{i+1}. {item}" for i, item in enumerate(question.answers)]
        )
        file.write(f"{answers}\n")
        file.write(
            f"Correct Answer: \n\t{question.correctAnswerIndex+1}. {question.answers[question.correctAnswerIndex]}\n"
        )
        file.write(f"Reason: {question.reason}\n")

        file.write(f"Citations: {question.citations}\n\n")
