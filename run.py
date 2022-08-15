import hydra, os
from rsb import RSB


@hydra.main(config_path="./conf/toy", config_name="config_25gauss")
def main(args):
    print('Directory: ' + os.getcwd())
    rsb = RSB(args)
    if args.mode == 'train':
        rsb.train()
    elif args.mode == 'test':
        rsb.test()


if __name__ == '__main__':
    main()