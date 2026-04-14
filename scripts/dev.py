if __name__ == "__main__":
    from scripts.sk.trainer import Trainer

    t = Trainer("deit")
    t.training_loop()

    from scripts.sk.inference.inference import Inference

    i = Inference("deit")
