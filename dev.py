if __name__ == "__main__":
    from sk.trainer import Trainer

    t = Trainer("deit")
    t.training_loop()

    from sk.inference.inference import Inference

    i = Inference("deit")