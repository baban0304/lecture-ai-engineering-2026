# 今後のテストで使う可能性があるため、ここに残しておく。
from app import ask


def test_chat_sends_follow_up_messages_in_the_same_session(monkeypatch, capsys):
    class Response:
        def __init__(self, text):
            self.text = text
            self.usage_metadata = type(
                "Usage",
                (),
                {"prompt_token_count": 12, "candidates_token_count": 34},
            )()
            self.model_version = "test-model"

    class Chat:
        def __init__(self):
            self.messages = []

        def send_message(self, message):
            self.messages.append(message)
            return Response(f"answer to {message}")

    class Chats:
        def __init__(self):
            self.session = Chat()

        def create(self, model):
            assert model == ask.MODEL
            return self.session

    class Client:
        def __init__(self):
            self.chats = Chats()

    client = Client()
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda api_key: client)
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat"])
    questions = iter(["first question", "follow-up", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask.main()

    assert client.chats.session.messages == ["first question", "follow-up"]
    output = capsys.readouterr().out
    assert output.index("[回答] answer to follow-up") < output.index("[入力トークン数] 12")
    assert "[出力トークン数] 34" in output
