"""""
Original file is located at
    https://colab.research.google.com/drive/1RZnCEyAwIaLcNeTOaybozPpAmBYW5JiB
"""

from google.cloud import dialogflow
import logging
import os
import requests
from os.path import join, dirname
from dotenv import load_dotenv

load_dotenv()
secret_key = os.environ.get("Gemini_youmna_key")
class Gemini:
    def __init__(self, model="gemini-1.0", prompt=None, api_key=secret_key):
        self.model = model
        self.api_key = api_key
        if prompt is None:
            self.prompt = [
                {
                    "role": "user",
                    "content": "",  # this will be replaced with your message
                }
            ]
        else:
            self.prompt = prompt

    def ask_gemini(self, message, session_id="12345", **query_kwargs):
        """
        * message: the text message that will be sent to Gemini for processing
        * session_id: unique session id
        * query_kwargs: Additional parameters to modify the request
        """
        url = f"https://dialogflow.googleapis.com/v2/projects/your_project_id/agent/sessions/{session_id}:detectIntent"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "queryInput": {
                "text": {
                    "text": message,
                    "languageCode": "en"
                }
            }
        }

        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            response_dict = response.json()
            result = response_dict["queryResult"]["fulfillmentText"]
            return result
        except requests.exceptions.RequestException as e:
            logging.error(f"Error: {e}")
            return None


class RephraseGemini(Gemini):
    def __init__(self, model="gemini-1.0", prompt=None, api_key="your_api_key"):
        super().__init__(model, api_key=api_key, prompt=prompt)
        if prompt is None:
            self.prompt = [
                {
                    "role": "system",
                    "content": "You are a rephrasing assistant. " \
                               "You are brilliant in creatively rephrasing whatever the user indicates you. " \
                               "When rephrasing, you can cleverly vary the grammatical structure of sentences without changing their overall meaning. " \
                               "Additionally, you can change words too, as long as the meaning of the sentence is preserved. " \
                               "Also, you do not miss any detail when rephrasing sentences."
                },
                {
                    "role": "user",
                    "content": "",  # this will be replaced with your message
                }
            ]
        else:
            self.prompt = prompt

    def ask_gemini_rephrase(self, prompt, session_id="12345", **query_kwargs):
        message = f"Rephrase the following prompt: {prompt}"
        return super().ask_gemini(message, session_id, **query_kwargs)


class MathGemini(Gemini):
    def __init__(self, model="gemini-1.0", prompt=None, api_key="your_api_key"):
        super().__init__(model, api_key=api_key, prompt=prompt)
        if prompt is None:
            self.prompt = [
                {
                    "role": "user",
                    "content": "Pretend you are an expert Math teacher and Python coder. \nCompare the following two solutions to a math problem: \"{problem}\"\n" +
                               "The Solution1 is: {sol1}\nSolution1 ends here. \n\n\nThe Solution2 is: {sol2}\nSolution2 ends here.\n" +
                               "Write a JSON string comparing the solutions. Think step by step and identify what is incorrect. Example:\n" +
                               "{{\n" +
                               "\"discussion\": \"Let's think step by step. First ... Therefore SolutionX seems to be correct, while SolutionY is likely incorrect\",\n" +
                               "\"better_solution_is\": \"1\"\n" +
                               "}}"
                }
            ]

    def ask_gemini_math(self, problem, sol1, sol2, session_id="12345", **query_kwargs):
        message = f"Pretend you are an expert Math teacher and Python coder. \nCompare the following two solutions to a math problem: \"{problem}\"\n" + \
                  f"The Solution1 is: {sol1}\nSolution1 ends here. \n\n\nThe Solution2 is: {sol2}\nSolution2 ends here.\n" + \
                  f"Write a JSON string comparing the solutions. Think step by step and identify what is incorrect. Example:\n" + \
                  "{{\n" + \
                  "\"discussion\": \"Let's think step by step. First ... Therefore Solution1 seems to be correct, while Solution2 is likely incorrect\",\n" + \
                  "\"better_solution_is\": \"1\"\n" + \
                  "}}"
        return super().ask_gemini(message, session_id, **query_kwargs)

    def save_output_to_file(self, output, file_name="output.txt"):
        """
        Saves the output to a text file
        """
        with open(file_name, "a") as file:
            file.write(output + "\n")
            file.write("=" * 50 + "\n")  # Adds a separator for clarity


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Replace with your API key
    api_key = "your_api_key"

    print("========= TESTING MATH GEMINI =========")
    math_gemini = MathGemini(api_key=api_key)
    problem = "Solve the equation 2x + 3 = 7"
    sol1 = "x = 2"
    sol2 = "x = 4"
    result = math_gemini.ask_gemini_math(problem, sol1, sol2)
    print(result)

    # Save output to a text file
    math_gemini.save_output_to_file(result)

    print("========= TESTING REPHRASE GEMINI =========")
    rephrase_gemini = RephraseGemini(api_key=api_key)
    problem = "If 3 boys each make 12 muffins for a bake sale, and 2 other girls are making 20 muffins each, how many total muffins will be for sale?"
    rephrase_result = rephrase_gemini.ask_gemini_rephrase(problem)
    print(rephrase_result)

    # Save output to a text file
    rephrase_gemini.save_output_to_file(rephrase_result)