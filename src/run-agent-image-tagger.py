
import os 
import yaml
import json 
import argparse
# ------------------------------------------------------

from dataclasses import dataclass, asdict
from openai import OpenAI
from agents import Agent, Runner
from PIL import Image

# ------------------------------------------------------

@dataclass
class AgentConfig:
    name: str
    model: str
    instructions: str

@dataclass
class UserConfig:
    role: str
    content: list


class ImageTaggerAgent:

    CFG_KEY_AGENT = "agent"
    CFG_KEY_USER = "user"

    def __init__(self, path_cfg: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

        fstream = open(path_cfg, "r")
        cfg = yaml.safe_load(fstream)
        fstream.close()
        self.agent_cfg = AgentConfig(**cfg[self.CFG_KEY_AGENT])
        self.user_cfg = UserConfig(**cfg[self.CFG_KEY_USER])

        self.agent = Agent(name=self.agent_cfg.name, model=self.agent_cfg.model, instructions=self.agent_cfg.instructions)

    def _form_user_message(self, file_id) -> dict:
        for content_shard in self.user_cfg.content:
            if content_shard["type"] == "input_image":
                content_shard["file_id"] = file_id
        return asdict(self.user_cfg)


    def tag_photo(self, path_image: str) -> dict:
        try:
            path_temporary_image = "out-temp.jpg"
            img = Image.open(path_image)
            img.thumbnail((1024,1024), Image.Resampling.LANCZOS)
            img.save(path_temporary_image, quality=90)

            file = self.client.files.create(file=open(path_temporary_image, "rb"), purpose="user_data")

            user_message = self._form_user_message(file.id)

            result = Runner.run_sync(self.agent, [user_message])
            output = json.loads(result.final_output)
            return output 
        except Exception as e:
            return None

# ------------------------------------------------------

def run(path_cfg: str, path_in: str, path_out_dir: str) -> bool:
    if not os.path.exists(path_in): return False
    if not os.path.exists(path_out_dir): return False
    if not os.path.isdir(path_out_dir): return False

    if os.path.isfile(path_in):
        if not path_in.endswith(".jpg"): return False

        agent = ImageTaggerAgent(path_cfg)
        tag_data = agent.tag_photo(path_in)

        if tag_data is None:
            print(f"[ERROR]: Bad response for {file}")
            return False
        
        file_name = os.path.basename(path_in).replace(".jpg", "")
        f = open(os.path.join(path_out_dir, file_name + ".json"), "w")
        json.dump(tag_data, f, ensure_ascii=False, indent=4)
        f.close()
        
        return True

    if os.path.isdir(path_in):
        fl = [f for f in os.listdir(path_in) if f.endswith(".jpg")]
        for file in fl:
            print(f"[LOG]: Processing {file}")
            path_img = os.path.join(path_in, file)
            agent = ImageTaggerAgent(path_cfg)
            tag_data = agent.tag_photo(path_img)

            if tag_data is None:
                print(f"[ERROR]: Bad response for {file} -> skipping")

            file_name = os.path.basename(file).replace(".jpg", "")
            f = open(os.path.join(path_out_dir, file_name + ".json"), "w")
            json.dump(tag_data, f, ensure_ascii=False, indent=4)
            f.close()

        return True

# ------------------------------------------------------

INFO = (
    "Generate tags for an image using OpenAI API"
    "API key should be set in the OPENAI_API_KEY environment variable"
    "-c <path> - path to the agent config file"
    "-i <path> - path to a file or directory to process"
    "-o <path> - path to a directoryu where generated tag JSON will be saved\n"
)

def parse_arguments():
    parser = argparse.ArgumentParser(description=INFO, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", type=str, help="path to the agent config file")
    parser.add_argument("-i", type=str, help="path to a file or directory to process")
    parser.add_argument("-o", type=str, help="path to a directoryu where generated tag JSON will be saved")
    args = parser.parse_args()
    return args.c, args.i, args.o

# ------------------------------------------------------

if __name__ == '__main__':
    path_cfg, path_in, path_out = parse_arguments()
    run(path_cfg, path_in, path_out)