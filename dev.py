from sk.dataset.new_chexpert import CheXpert
from sk.trainers.trainer import Trainer

if __name__ == "__main__":
    from sk.trainers.trainer import Trainer

    t = Trainer("efficient")
    t.training_loop()