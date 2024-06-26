import logging
import sys
import os
import json
import pandas as pd
from configData import outputFolder
from transcriptLoader import retrieveTranscript
from topicExtractor import retrieveTopics
from BERTopicQuestionGenerator import retrieveBERTopicQuestions
from LangChainQuestionGenerator import retrieveLangChainQuestions


class QuestionData:
    """
    Represents a collection of questions generated from raw question data.

    Args:
        config (object): Configuration object containing generation model information.
        rawQuestionData (object): Raw question data used for generating the questions.

    Attributes:
        config (object): Configuration object containing generation model information.
        questions (dict): Dictionary to store the generated questions.

    Methods:
        processBERTopicQuestions(BERTdata): Process the raw question data using the BERTopic generation model.
        processLangChainQuestions(LangChainData): Process the raw question data using the LangChain generation model.
        makeDF(): Convert the generated questions into a pandas DataFrame.
        saveToFile(): Save the generated questions to a file.
        printQuestions(): Print the generated questions.

    """
    def __init__(self, config, rawQuestionData):
        """
        Initializes a QuestionGenerator object.

        Args:
            config (Config): The configuration object containing generation model settings.
            rawQuestionData (list): The raw question data used for generating questions.

        Returns:
            None
        """
        self.config = config
        self.questions = {}

        if self.config.generationModel == "BERTopic":
            self.processBERTopicQuestions(rawQuestionData)
        elif self.config.generationModel == "LangChain":
            self.processLangChainQuestions(rawQuestionData)

    def processBERTopicQuestions(self, BERTdata):
        """
        Process BERT topic questions.

        Args:
            BERTdata (object): The BERT data object.

        Returns:
            None

        Raises:
            None
        """
        for index, row in BERTdata.responseInfo.iterrows():
            origStart = None
            if "Original Start" in row and type(row["Original Start"]) == type(
                row["Start"]
            ):
                origStart = row["Original Start"]

            response = row["Response Data"]
            response = response.strip("` \n")
            if response.startswith("json"):
                response = response[4:]

            try:
                parsedResponse = json.loads(response)
                question = parsedResponse["question"]
                answers = parsedResponse["answers"]
                if parsedResponse["correct"] in answers:
                    correctIndex = answers.index(parsedResponse["correct"])
                else:
                    correctIndex = "Correct answer not found in answers."
                reason = parsedResponse["reason"]

            except json.JSONDecodeError as e:
                logging.warn(
                    f"Error decoding JSON: {e} for received response: {parsedResponse}"
                )
                logging.warn(
                    f"Question {index+1} for topic: {row['Topic Title']} not generated."
                )

                question = None
                answers = None
                correctIndex = None
                reason = None

            self.questions[index] = {
                "Start": row["Start"],
                "End": row["End"],
                "Topic": row["Topic Title"],
                "Keywords": row["Words"],
                "Original Start": origStart,
                "Question": question,
                "Answers": answers,
                "Correct Answer Index": correctIndex,
                "Reason": reason,
            }

    def processLangChainQuestions(self, LangChainData):
        """
        Process the language chain questions.

        Args:
            LangChainData: The data containing the language chain information.

        Returns:
            None
        """
        self.questions = LangChainData.responseInfo

    def makeDF(self):
        """
        Converts the questions dictionary into a pandas DataFrame and performs some data transformations.

        Returns:
            pandas.DataFrame: The transformed DataFrame containing the questions data.
        """
        questionDF = pd.DataFrame.from_dict(self.questions).T
        questionDF["Start"] = pd.to_datetime(questionDF["Start"]).dt.strftime("%H:%M:%S")
        questionDF["End"] = pd.to_datetime(questionDF["End"]).dt.strftime("%H:%M:%S")
        questionDF['Answers'] = questionDF['Answers'].apply(lambda answers: "\n".join([f"{i+1}. {item}" for i, item in enumerate(answers)]))
        questionDF['Correct Answer'] = questionDF['Correct Answer Index'].apply(lambda option: f"Option {option+1}")
        del questionDF['Correct Answer Index']

        questionDF['Model'] = self.config.generationModel
        questionDF = questionDF[['Model', 'Question', 'Topic', 'Start', 'End', 'Answers', 'Correct Answer', 'Reason']]

        questionDF = questionDF.set_index(['Model', 'Question'])
        questionDF["Is this question:\n Relevant to the Transcript"] = ""
        questionDF["Is this question:\n Good Quality"] = ""
        questionDF["Is this question:\n Useful"] = ""
        return questionDF

    def saveToFile(self):
        """
        Saves the question data to a file.

        The method creates a directory for the output file if it doesn't exist.
        Then it saves the question data to a text file with a specific format.

        Raises:
            OSError: If the creation of the directory or saving the question data fails.
            Exception: If there is an error while saving the question data.

        Returns:
            None
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
                writeDataToFile(file, self.config.videoToUse, self.questions)
            logging.info(f"Question Data saved to file: {questionSavePath}")
        except OSError:
            logging.warn(f"Failed to save question data to file: {questionSavePath}")
        except Exception as e:
            logging.warn(f"Failed to save question data to file: {questionSavePath}")
            logging.warn(f"Error: {e}")

    def printQuestions(self):
        """
        Prints the questions stored in the 'questions' attribute.

        Each question is logged with its index, topic, transcript segment, insertion point,
        question text, answers, correct answer, and reason.

        If a question is not generated, it is logged as such.
        """
        for index in self.questions:
            question = self.questions[index]

            logging.info("\n---------------------------------------")
            logging.info(f"Question {index+1}")
            logging.info(f"Topic: {question['Topic']}")
            logging.info(
                f"Transcipt Segment: {question['Start'].strftime('%H:%M:%S')}"
                + f" - {question['End'].strftime('%H:%M:%S')}"
            )

            logging.info(f"Insertion Point: {question['End'].strftime('%H:%M:%S')}")

            if question["Question"]:
                logging.info(f"\nQuestion: {question['Question']}")
                logging.info(
                    "\nAnswers:"
                    + "\n\t".join(
                        [f"{i+1}. {item}" for i, item in enumerate(question["Answers"])]
                    )
                )
                logging.info(
                    "\nCorrect Answer:\n\t"
                    + f"\n\t{question['Correct Answer Index']+1}. " 
                    + f"{question['Answers'][question['Correct Answer Index']]}\n"
                )
                logging.info(f"Reason: {question['Reason']}")
            else:
                logging.info("Question not generated.")


def retrieveQuestions(
    config, topicModeller=None, videoData=None, overwrite=False, saveToFile=True
):
    """
    Retrieves questions based on the specified generation model.

    Args:
        config (object): Configuration object containing generation model information.
        topicModeller (object, optional): Topic modeller object. Defaults to None.
        videoData (object, optional): Video data object. Defaults to None.
        overwrite (bool, optional): Flag indicating whether to overwrite existing data. Defaults to False.
        saveToFile (bool, optional): Flag indicating whether to save the question data to a file. Defaults to True.

    Returns:
        object: Question data object.
    """
    if config.generationModel == "BERTopic":
        rawQuestionData = retrieveBERTopicQuestions(
            config,
            topicModeller=topicModeller,
            videoData=videoData,
            overwrite=overwrite,
        )
    elif config.generationModel == "LangChain":
        rawQuestionData = retrieveLangChainQuestions(
            config, videoData=videoData, overwrite=overwrite
        )
    else:
        logging.error(
            f"Invalid generation model specified: {config.generationModel}, valid options are 'BERTopic' and 'LangChain'."
        )
        sys.exit("Invalid generation model specified. Exiting...")

    questionData = QuestionData(config, rawQuestionData)

    if saveToFile:
        questionData.saveToFile()
    return questionData


def processCaptions(config, overwrite=False, saveToFile=True):
    """
    Process captions to retrieve transcript, topics, and questions for a given video.

    Args:
        config (object): Configuration object containing video information.
        overwrite (bool, optional): Flag indicating whether to overwrite existing data. Defaults to False.
        saveToFile (bool, optional): Flag indicating whether to save the questions to a file. Defaults to True.

    Returns:
        None
    """
    logging.info(f"Retrieving Transcript for {config.videoToUse}...")
    videoData = retrieveTranscript(config, overwrite=overwrite)

    topicModeller = None
    if config.generationModel == "BERTopic":
        logging.info(f"Retrieving Topics for {config.videoToUse}...")
        topicModeller = retrieveTopics(config, videoData, overwrite=overwrite)
        topicModeller.printTopics()

    logging.info(f"Retrieving Questions for {config.videoToUse}...")
    questionData = retrieveQuestions(
        config,
        topicModeller=topicModeller,
        videoData=videoData,
        overwrite=overwrite,
        saveToFile=saveToFile,
    )
    questionData.printQuestions()


def writeDataToFile(file, videoName, questions):
    """
    Writes data to a file.

    Args:
        file (file object): The file object to write the data to.
        videoName (str): The name of the video or parent folder.
        questions (dict): A dictionary containing the questions and related information.

    Returns:
        None
    """
    file.write(f"Video Name / Parent Folder: {videoName}\n")
    file.write("--------------------------------------------------\n")
    for index in questions:
        question = questions[index]

        file.write("\n---------------------------------------\n")
        file.write(f"Question {index+1}\n")
        file.write(f"Topic: {question['Topic']}\n")

        if "Keywords" in question:
            file.write(f"Keywords: {question['Keywords']}\n\n")

        durationMin, durationSec = divmod(
            (question["End"] - question["Start"]).total_seconds(), 60
        )
        file.write(
            f"Transcript Segment: {question['Start'].strftime('%H:%M:%S')}"
            + f" - {question['End'].strftime('%H:%M:%S')}\n"
        )
        file.write(
            f"Duration: {int(durationMin)} minutes & {int(durationSec)} seconds\n"
        )

        if "Original Start" in question and question["Original Start"]:
            trueDurationMin, trueDurationSec = divmod(
                (question["End"] - question["Original Start"]).total_seconds(), 60
            )
            file.write(
                f"\tTruncated from original duration of {int(trueDurationMin)} minutes & {int(trueDurationSec)} seconds\n"
            )

        file.write(f"Insertion Point: {question['End'].strftime('%H:%M:%S')}\n")

        if question["Question"]:
            file.write(f"\nQuestion: {question['Question']}\n")
            file.write(
                "\nAnswers:\n\t"
                + "\n\t".join(
                    [f"{i+1}. {item}" for i, item in enumerate(question["Answers"])]
                )
            )
            file.write("\nCorrect Answer:"
                    + f"\n\t{question['Correct Answer Index']+1}. " 
                    + f"{question['Answers'][question['Correct Answer Index']]}\n")
            file.write(f"Reason: {question['Reason']}\n")
        else:
            file.write("\nQuestion not generated.\n")