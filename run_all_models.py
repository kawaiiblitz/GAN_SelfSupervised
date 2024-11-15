import os


def execute(cmd: str, log: bool = True) -> None:
    """
    Executes a command in the terminal.
    """
    if log:
        print(cmd)
    
    code = os.system(cmd)
    if code != 0:
        raise Exception(f"failed with error code {code}")


def run(mode: str, model_path: str | None = None) -> None:
    """
    Runs the training or evaluation process for a given mode and optional model path.

    Args:
        mode (str): The mode to run, which can be 'baseline' for running the baseline model or 'ft' for fine-tuning.
        model_path (str | None): The path to the pretrained model if mode is 'ft'. Default is None.

    Description:
        - For 'baseline' mode, it runs the baseline model training.
        - For 'ft' mode, it performs fine-tuning on a pretrained model over a range of dataset percentages.
    """
    print(f"running mode {mode}, model {model_path} ...")
    if mode == "baseline":
        execute("python main.py --mode baseline")
    else: 
        start, end, step = 1.0, 0.1, 0.1
        n = int((start - end) / step)

        for i in range(n + 1):
            pct = round(start - i * end, 1)
            execute(f"python main.py --mode ft --model_path {model_path} --percent {pct}")


if __name__ == "__main__":
    # baseline CNN without any pre-trained model
    run("baseline")
    
    # ablation tests on the pretrained jigsaw GAN (9 piece)
    run("ft", "./pretrained_models/GAN_jigsaw9_64_200e/discriminator.pth")
    
    # ablation tests on the pretrained GAN (no jigsaw)
    run("ft", "./pretrained_models/GAN_jigsaw0_64_200e/discriminator.pth")
