import argparse
from famie.scripts import start_app

def command_start(args):
    print("Starting a new session...")
    start_app.main(args)


def main():
    parser = argparse.ArgumentParser(
        description="FAMIE: A Fast Active Learning Framework for Multilingual Information Extraction.")
    subparsers = parser.add_subparsers()
    parser_start = subparsers.add_parser("start", help="Subparser for creating a new session.")
    parser_start.add_argument("--selection",
                              type=str,
                              default="mnlp",
                              help="Data selection strategy",
                              choices=['mnlp', 'bertkm', 'badge', 'random'])
    parser_start.add_argument("--port",
                              type=str,
                              default="9000",
                              help="Port specification")
    parser_start.add_argument("--target_embedding",
                              type=str,
                              default='xlm-roberta-base',
                              help="Pretrained language model for the main model, default='xlm-roberta-large'",
                              choices=['xlm-roberta-base', 'xlm-roberta-large'])
    parser_start.add_argument("--proxy_embedding",
                              type=str,
                              default='nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large',
                              help="Pretrained Language Model for the proxy model, default='nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large'",
                              choices=['nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large',
                                       'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large'])
    parser_start.set_defaults(handler=command_start)

    '''
    if args.action == 'run':
        start_app.main()
    elif args.action == 'uninstall':
        uninstall_app.main(config)
    '''
    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)


if __name__ == "__main__":
    main()
