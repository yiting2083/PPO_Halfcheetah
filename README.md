You need to set up your own wandb account to see the plots. How to use wandb? It is easy to know how to set up it by ask any language models or from google search.
It means here(line 26 and line 30 in ppo.py) you need to change to your own key


class WandB:
    os.environ["WANDB_API_KEY"] = "your own key"
    project_name = "ppocheetah"
os.environ["WANDB_API_KEY"] = "your own key"
wandb.init(project=WandB.project_name)
