"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保って対話する

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    response = generate_response(question)
    print_usage(response)
    return response.text


def generate_response(question: str):
    client = genai.Client(api_key=get_api_key())
    return client.models.generate_content(model=MODEL, contents=question)


def print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(f"[入力トークン数] {usage.prompt_token_count}")
        print(f"[出力トークン数] {usage.candidates_token_count}")
        print(f"[model_version] {response.model_version}")
    else:
        print("[トークン数] 利用量情報なし")


def chat() -> None:
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL)
    print("会話を開始します。終了するには /exit または Ctrl-D を入力してください。")

    while True:
        try:
            question = input("[質問] ")
        except EOFError:
            print()
            return

        if question.strip().lower() in {"/exit", "/quit"}:
            return
        if not question.strip():
            continue

        response = session.send_message(question)
        print(f"[回答] {response.text}")
        print_usage(response)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って対話する")
    args = parser.parse_args()

    if args.chat:
        print(f"[model] {MODEL}")
        try:
            chat()
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        return

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        response = generate_response(args.question)
        print(response.text)
        print_usage(response)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
