from tqdm import tqdm

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--file',
        default='./output.jsonl',
        type=str,
        metavar='PATH',
        help='data file to dump (default: ./output.jsonl)',
    )

    parser.add_argument(
        '-o', '--output',
        default='./output-formatted.txt',
        type=str,
        metavar='PATH',
        help='output file (default: ./output-formatted.txt)'
    )
    return parser.parse_args()

def main(args):
    with open(args.file, 'r') as f:
        data = [ json.loads(line) for line in f ]

    with open(args.output, 'w') as f:
        for chat_id, chat in enumerate(tqdm(data)):
            f.write(f'Chat {chat_id}\n')
            for idx, text in enumerate(chat['dialog']):
                if idx % 2:
                    f.write(f'{"Bot: ": ^11}{text}\n')
                else:
                    f.write(f'{"Simulator: ": ^11}{text}\n')

            f.write('\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)