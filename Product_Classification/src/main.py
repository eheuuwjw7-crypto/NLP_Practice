from argparse import ArgumentParser

if __name__ == "__main__":
    # 定义参数解析器
    parser = ArgumentParser(usage="python main.py [preprocess | train | predict | evaluate | serve]")
    parser.add_argument('action', choices=["preprocess", "train", "predict", "evaluate", "serve"])
    arg = parser.parse_args()
    action = arg.action

    match action:
        case "preprocess":
            from src.preprocess.preprocese import preprocess
            preprocess()
        case "train":
            from src.runner.train import train
            train()
        case "predict":
            from src.runner.predict import predict
            predict()
        case "evaluate":
            from src.runner.evaluate import evaluate
            evaluate()
        case "serve":
            from src.web.app import serve
            serve()
        case _:
            print("请输入 train、predict 或 evaluate")