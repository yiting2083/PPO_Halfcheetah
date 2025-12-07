You need to set up your own wandb account to see the plots. It means here you need to change to your own key
How to use wandb? It is easy to know how to set up it by ask any language models or from google search.

class WandB:
    os.environ["WANDB_API_KEY"] = "your own key"
    project_name = "ppocheetah"

set_seed(45)
os.environ["WANDB_API_KEY"] = "your own key"
wandb.init(project=WandB.project_name)
