import os
from playsound import playsound
import sys
from openai import OpenAI
import api

def setup_client():
    api_key =  f"{api.key}"
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    return OpenAI(api_key=api_key)


class ChatBot:
    def _init_(self, model="gpt-4.1-mini"):
        self.client = setup_client()
        self.model = model
        self.system_prompt = """Você receberá uma sequência de texto formada apenas por letras, sem espaços ou pontuação. Sua tarefa é:

1. Reescrever o texto como uma frase coerente, adicionando os espaços e a pontuação adequados.
2. Corrigir eventuais erros de digitação ou ortografia que possam estar presentes.
3. Garantir que a frase final fique clara, gramaticalmente correta e soe natural.

Exemplo de entrada:
esteeumtesteparasuashabilidades

Exemplo de saída:
Este é um teste para suas habilidades."""

        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def send_message(self, message):
        self.conversation_history.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                max_tokens=1000,
                temperature=0.3
            )
            assistant_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("Conversation history cleared (system prompt maintained).")

    def show_prompt(self):
        print("\nCurrent system prompt:")
        print("-" * 40)
        print(self.system_prompt)
        print("-" * 40)
        
def main():
    flag = 1
    text = ""
    i = 0
    
    bot = ChatBot()
    print("Comandos: 'sair/q', 'clear', 'prompt'")
    while True:
        try:
            user_input = input("\nTexto para formatar: ").strip()
            if user_input.lower() in ['sair','q']:
                print("Goodbye! / Tchau!")
                break
            elif user_input.lower() in ['clear', 'limpar']:
                bot.clear_history()
                continue
            elif user_input.lower() in ['prompt', 'mostrar']:
                bot.show_prompt()
                continue
            elif not user_input:
                continue
            print("Formatted: ", end="", flush=True)
            response = bot.send_message(user_input)
            print(response)
            os.system(f"echo \"{response}\" | .\Codigo\Piper\piper.exe -m .\Codigo\Piper\Voices\pt_BR-faber-medium.onnx -f .\Codigo\output{i}.wav")
            playsound(f'.\Codigo\output{i}.wav')
            i += 1
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! / Tchau!")
            break
        except EOFError:
            print("\n\nGoodbye! / Tchau!")
            break
                    
if __name__ == "__main__":
    main()
