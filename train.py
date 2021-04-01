import argparse

from trainer import WassersteinTrainer

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--__epochs__", type=int, default=500, help="Number of Epochs for Training")
    PARSER.add_argument("--__steps_per_epoch__", type=int, default=10000, help="Number of Batches per Epoch")
    PARSER.add_argument("--__checkpoint_interval__", type=int, default=1, help="Interval Between Model Checkpoints")
    PARSER.add_argument("--__width__", type=int, default=512, help="Width of inputs")
    PARSER.add_argument("--__height__", type=int, default=512, help="Height of inputs")
    PARSER.add_argument("--__flist__", type=str, default="./Data/Images", help="Images Dataset Filepath")
    PARSER.add_argument('--__gen_masks__', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Generate Masks on the fly")
    PARSER.add_argument("--__mask_type__", type=str, default="bbox", help="Generate free-form or bbox masks")
    PARSER.add_argument("--__box_count__", type=int, default=10, help="Number of bbox masks")
    PARSER.add_argument("--__mlist__", type=str, default="./Data/Masks", help="Mask Dataset Filepath")
    PARSER.add_argument("--__savepath__", type=str, default="./Models", help ="Model Saver Path")
    PARSER.add_argument("--__samplepath__", type=str, default="./Samples", help ="Model Output Saver Path")
    PARSER.add_argument("--__gexists__", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Load Pre-Existing Generator")
    PARSER.add_argument("--__gpath__", type=str, default="./Models/Generator", help="Pre-Existing Generator Filepath")
    PARSER.add_argument("--__lrg__", type=float, default=1e-4, help="Generator Learning Rate")
    PARSER.add_argument("--__dexists__", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Load Pre-Existing Discriminator")
    PARSER.add_argument("--__dpath__", type=str, default="./Models/Discriminator",
                        help="Pre-Existing Discriminator Filepath")
    PARSER.add_argument("--__lrd__", type=float, default=4e-4, help="Discriminator Learning Rate")
    PARSER.add_argument("--__activation__", type=str, default="elu", help="Model Activation")
    PARSER.add_argument("--__lambda_l1__", type=float, default=100, help="Parameter of L1Loss")
    PARSER.add_argument("--__lambda_gan__", type=float, default=1, help="Parameter of Gan")
    PARSER.add_argument("--__lambda_perceptual__", type=float, default=10,
                        help="Parameter of FML1Loss (perceptual loss)")

    OPT = PARSER.parse_args()

    TRAINING = WassersteinTrainer(OPT)

