import sys
import os

gpt_conf = {'GPT3-Small' : {'n_layers' : 12, 'd_model' : 768, 'n_head':12, 'd_head' : 64},
            'GPT3-Medium' : {'n_layers' : 24, 'd_model' : 1024, 'n_head':16, 'd_head' : 64},
            'GPT3-Large' : {'n_layers' : 24, 'd_model' : 1536, 'n_head':16, 'd_head' : 96},
            'GPT3-XL  ' : {'n_layers' : 24, 'd_model' : 2048, 'n_head':24, 'd_head' : 128},
            'GPT3-2.7B' : {'n_layers' : 32, 'd_model' : 2560, 'n_head':32, 'd_head' : 80},
            'GPT3-6.7B' : {'n_layers' : 32, 'd_model' : 4096, 'n_head':32, 'd_head' : 128},
            'GPT3-13B' : {'n_layers' : 40, 'd_model' : 5140, 'n_head':40, 'd_head' : 128},
            'GPT3-175B(GPT3)' : {'n_layers' : 96, 'd_model' : 12288, 'n_head':96, 'd_head' : 128}}
n_vocab = 50257
n_ctx = 2048

def gpt_params_spend() :
    # parameters
    for key, conf in gpt_conf.items() :
        pos_emb = n_ctx * conf['d_model']
        word_emb = n_vocab * conf['d_model'] 
        attn = 4 * conf['n_layers'] * conf['n_head'] * conf['d_head'] * conf['d_model']
        n_ff = 4 * conf['d_model']
        ffn = conf['n_layers'] * (2 * n_ff * conf['d_model'] + n_ff + conf['d_model'])
        layer_norm = (conf['n_layers'] * 2 + 1)* conf['d_model']
        total = pos_emb + word_emb + attn + ffn + layer_norm
        print(("{11}\ttotal:{10:,d}\tpos_emb:{0:,d}({1:.2f}%)\tword_emb:{2:,d}({3:.2f}%)"
            + "\tattn:{4:,d}({5:.2f}%)\tffn:{6:,d}({7:.2f}%)\tlayer_norm:{8:,d}({9:.2f}%)").format(
            pos_emb, float(pos_emb)/total * 100, 
            word_emb, float(word_emb)/total * 100, 
            attn, float(attn)/total * 100, 
            ffn, float(ffn)/total * 100, 
            layer_norm, float(layer_norm)/total, total, key))

def gpt_computation_spend() : 
    # TODO: calculate tflops of forward propagation of GPT
    # https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py
    for key, conf in gpt_conf.items() : 
        attn = n_ctx * n_ctx 
    pass

if __name__ == "__main__" :
    gpt_params_spend()
    gpt_computation_spend()