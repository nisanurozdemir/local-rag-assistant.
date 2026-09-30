from foundry_local_sdk import Configuration, FoundryLocalManager
import inspect

config = Configuration(app_name="local_rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

chat_model = manager.catalog.get_model_variant("qwen2.5-1.5b-instruct-generic-cpu:4")
chat_model.download(lambda p: None)
chat_model.load()
chat_client = chat_model.get_chat_client()

print(inspect.signature(chat_client.complete_streaming_chat))
print(inspect.signature(chat_client.complete_chat))
