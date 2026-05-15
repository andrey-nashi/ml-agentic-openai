
import os 
import yaml

# ------------------------------------------------------
from dataclasses import dataclass, asdict
from agents import Agent, Runner, function_tool

# ------------------------------------------------------

class Database:

    def __init__(self, weather_db, stock_db):
        self.weather_db = None
        self.stock_db = None

        self.weather_db = weather_db
        self.stock_db = stock_db

        print(self.weather_db)
        print(self.stock_db)

    def get_weather(self, location_name: str) -> str:
        print(f"[LOG]: Calling function get_weather for param >{location_name}<")
        location_name = location_name.lower()
        if location_name in self.weather_db:
            print(f"[LOG]: returning {self.weather_db[location_name]}")
            return str(self.weather_db[location_name])
        return "unknown"
    
    def get_stock(self, stock_name: str) -> str:
        print(f"[LOG]: Calling function get_stock for param >{stock_name}<")
        stock_name = stock_name.lower()
        if stock_name in self.stock_db:
            print(f"[LOG]: returning {self.stock_db[stock_name]}")
            return str(self.stock_db[stock_name])
        return "unknown"

@dataclass
class AgentConfig:
    name: str
    model: str
    instructions: str

path_cfg = "./cfg/agent-function-call.yaml"
CFG_KEY_WEATHER_DB = "weather_db"
CFG_KEY_SOTCK_DB = "stock_db"
CFG_KEY_AGENT = "agent"

fstream = open(path_cfg, "r")
cfg = yaml.safe_load(fstream)
fstream.close()

agent_cfg = AgentConfig(**cfg[CFG_KEY_AGENT])
db = Database(cfg[CFG_KEY_WEATHER_DB], cfg[CFG_KEY_SOTCK_DB])

# ------------------------------------------------------

@function_tool
def get_weather(location_name: str) -> str:
    """This function provides the weather information based on the location name"""
    return db.get_weather(location_name)

@function_tool
def get_stock(stock_name: str) -> str:
    """This function provides stock information based on the company name"""
    return db.get_stock(stock_name)

# ------------------------------------------------------

agent = Agent(
    name=agent_cfg.name,
    model=agent_cfg.model,
    instructions=agent_cfg.instructions,
    tools=[
        get_weather,
        get_stock
    ]
)

def main():

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        if not user_input:
            continue

        try:
            result = Runner.run_sync(agent, user_input)
            print(f"Agent: {result.final_output}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()