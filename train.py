from trainer import training
import argparse
import sympy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='dataloader num_workers (default: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist',
                                                                 'cifar10',
                                                                 'cifar100'],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--EncryIP', action='store_true', default=False,
                        help='learning in enncrypted environment or not')
    parser.add_argument('--random', action='store_true', default=False,
                        help='learning in enncrypted random environment or not')
    parser.add_argument('--fake', action='store_true', default=False,
                        help='learning in enncrypted random environment or not')
    parser.add_argument('--p', type=int, default=11,
                        help='Encryption base parameters, which should be greater than the number of classes (default: 11, should be prime)')

    args = vars(parser.parse_args())
    if args['EncryIP']:
        if sympy.isprime(args['p']) == False:
            print("ERROR：p must be prime！")
            return
        if args['dataset'] == 'cifar100':
            args['p'] = 101
    elif args['random']:
        if sympy.isprime(args['p']) == False:
            print("ERROR：p must be prime！")
            return
        if args['dataset'] == 'cifar100':
            args['p'] = 101
    elif args['fake']:
        if sympy.isprime(args['p']) == False:
            print("ERROR：p must be prime！")
            return
        if args['dataset'] == 'cifar100':
            args['p'] = 101

    training(args)


if __name__ == "__main__":
    main()
