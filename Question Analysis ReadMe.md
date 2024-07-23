# Instructions for Analysis and Evaluation of Questions

This ReadMe outlines the steps needed to evaluate the generated Questions for a given transcript. 
The Excel file that follows the naming format `Question Analysis - ****.xlsx` will be the main file used. 

For every transcript, two generation models: `LangChain` and `BERTopic` are used to generate 4 questions each, for a total of 8 questions per transcript. 
Each Question can be evaluated across three categories:
1. Relevance - Is this question relevant to the topics covered in the transcript? 
2. Quality - Is this question of suitable quality? This is a more subjective determination of whether the question makes sense, and if the answers and reasoning are suffiecient.
3. Usefulness - Is this question useful in context of ensuring a viewer is paying attention to the video, and has understood the material presented?

A suitable evaluation schema would be for a scale of 1 to 5 to be used, where 1 is the lowest score, and a 5 is the highest score for any given question. There are 3 columns corresponding to each question in the Excel file where a score for each question can be typed in.

Each subfolder consists of 4 files: 
1. The Transcript .srt file used,
2. 2 .txt files for each of the questions generated from each generation model,
3. An Excel file to be used for evaluating the generated Questions.

The Excel file consists of most information that would be useful in evaluating each question including: 
1. Model
2. Question
3. Start & End Times
4. Possible Answers
5. Reasoning
6. 3 Evaluation questions
7. Transcript text used for generating the question

### TL,DR: Use the Excel file to rate each Question (4 questions each for 2 generation models) generated from 1 to 5 across 3 Evalutation Criteria.