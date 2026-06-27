import os
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
llm_name = "gemini-2.5-flash" 
client = genai.Client(api_key=google_api_key)

# response = client.models.generate_content(
#     model= llm_name,
#     contents="Who is Bhagat Singh ?"
# )

# print(response.text)



# creating our own agent
class Agent:
    def __init__(self, system="You are a helpful Agent"):
        self.system = system
        self.messages = []

    def __call__(self, message):
        self.messages.append({"role": "user", "parts": [{"text": message}]})
        result = self.execute()
        self.messages.append({"role": "model", "parts": [{"text": result}]})
        return result

    def execute(self):
        response = client.models.generate_content(
            model=llm_name,
            contents=self.messages,
            config={
                "system_instruction": self.system,
                "thinking_config": {"include_thoughts": True},
            },
        )
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                print(f"Thought: {part.text}\n")
        return response.text
    

# Prompt - how agent should behave 
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

planet_mass:
e.g. planet_mass: Earth
returns the mass of a planet in the solar system

Example session:

Question: What is the combined mass of Earth and Mars?
Thought: I should find the mass of each planet using planet_mass.
Action: planet_mass: Earth
PAUSE

You will be called again with this:

Observation: Earth has a mass of 5.972 × 10^24 kg

You then output:

Answer: Earth has a mass of 5.972 × 10^24 kg

Next, call the agent again with:

Action: planet_mass: Mars
PAUSE

Observation: Mars has a mass of 0.64171 × 10^24 kg

You then output:

Answer: Mars has a mass of 0.64171 × 10^24 kg

Finally, calculate the combined mass.

Action: calculate: 5.972 + 0.64171
PAUSE

Observation: The combined mass is 6.61371 × 10^24 kg

Answer: The combined mass of Earth and Mars is 6.61371 × 10^24 kg
""".strip()

# up until here, we have the agent, and we have the guidelines for it,
# but we need to give it tools to perform the required actions 


# Implement the function actions (the tools)
def calculate(what):
    return eval(what)


def planet_mass(name):
    masses = {
        "Mercury": 0.33011,
        "Venus": 4.8675,
        "Earth": 5.972,
        "Mars": 0.64171,
        "Jupiter": 1898.19,
        "Saturn": 568.34,
        "Uranus": 86.813,
        "Neptune": 102.413,
    }
    return f"{name} has a mass of {masses[name]} × 10^24 kg"


# some sort of map for out agent to work on and what actions to take 
known_actions = {"calculate": calculate, "planet_mass": planet_mass}

action_re = re.compile(r"^Action: (\w+): (.+)$", re.MULTILINE)


def query(question, max_turns=10):
    agent = Agent(system=prompt)
    next_prompt = question
    for _ in range(max_turns):
        result = agent(next_prompt)
        print(result)
        actions = action_re.findall(result)
        if not actions:
            return result
        action, action_input = actions[0]
        if action not in known_actions:
            raise ValueError(f"Unknown action: {action!r}")
        observation = known_actions[action](action_input.strip())
        print(f"Observation: {observation}\n")
        next_prompt = f"Observation: {observation}"
    return result


# query("What is the combined mass of Earth, Mars and Mercury?")

# Function to handle the interactive query
def query_interactive():
    bot = Agent(prompt)
    max_turns = int(input("Enter the maximum number of turns: "))
    i = 0

    while i < max_turns:
        i += 1
        question = input("You: ")
        result = bot(question)
        print("Bot:", result)

        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                print(f"Unknown action: {action}: {action_input}")
                continue
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = f"Observation: {observation}"
            result = bot(next_prompt)
            print("Bot:", result)
        else:
            print("No actions to run.")
            break


if __name__ == "__main__":
    query_interactive()


