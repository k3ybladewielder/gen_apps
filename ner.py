from transformers import pipeline
import gradio as gr

get_completion = pipeline("ner", model="dslim/bert-base-NER")

def merge_tokens(tokens):
    '''
    WHAT: Faz um loop entre os tokens para concatenar os tokens 
    com entidades I-* (intermediate token) aos B-* (begining token). 
    
    '''
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # Se a lista merged_tokens não estiver vazia.
            # o token atual for um token intermediário (começa com 'I-').
            # a entidade do último token processado terminar com a mesma entidade do token atual, excluindo o prefixo 'I-'.
            
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:            
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    """
    WHAT: Aplica a task de NER com o hugginface pipeline e concatena os tokens com a mesma entidade.
    RETURN: retorna um dicionário com o token e suas entidades.
    """
    output = get_completion(input) 
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Andrew, I'm building DeeplearningAI and I live in California", "My name is Poli, I live in Vienna and work at HuggingFace"])

demo.launch()
