import argparse


def train_parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for original images.")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='amazon_men',
                        help='dataset path: amazon_men, amazon_women, amazon_beauty')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--reg', type=int, default=1.0)
    parser.add_argument('--lr', type=int, default=0.05)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--window', type=int, default=0)

    # parser.add_argument('--defense', type=int, default=0)  # 0 --> no defense mode, 1 --> defense mode
    # parser.add_argument('--model_dir', type=str, default='free_adv')
    # parser.add_argument('--model_file', type=str, default='model_best.pth.tar')

    return parser.parse_args()
