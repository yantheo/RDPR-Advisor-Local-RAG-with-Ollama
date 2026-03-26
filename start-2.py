import ollama

#response = ollama.list()

# Response without streaming
# res = ollama.chat(
#     model="llama3.2",
#     messages=[{
#         "role": "user",
#         "content": "Why find a job a software engineer is challenging?",
#     }]
# )
# print(res["message"]["content"])


# Response without with streaming
# res = ollama.chat(
#     model="llama3.2",
#     messages=[{
#         "role": "user",
#         "content": "Why the ocean is so salty?",
#     }],
#     stream=True
# )
# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)

# Show all the info about LLM
# res= ollama.generate(
#     model="llama3.2",
#     prompt="Why not buy a house?"
# )

# print(ollama.show("llama3.2"))

# # 1. On définit les composants séparément
# model_name = 'knowitall'
# base_model = 'llama3.2'
# system_prompt = "You are a very smart assistant who knows everything about sport betting. You are very succinct and informative."

# # 2. La nouvelle syntaxe pour créer un modèle
# # On utilise 'from_' pour indiquer le modèle de base
# ollama.create(
#     model=model_name,
#     from_=base_model,
#     system=system_prompt,
#     parameters={
#         'temperature': 0.1
#     }
# )

# # 3. Utilisation
# print(f"Modèle {model_name} créé. Test en cours...")
# res = ollama.generate(model=model_name, prompt="Which kind of bet is more successful and why?")
# print(res["response"])


ollama.delete("knowitall")