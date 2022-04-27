from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from datasets import load_dataset


def make_plots(output, n_tokens):
    att1_all_layers = [[[] for j in range(n_tokens)] for i in range(n_tokens)]
    att2_all_layers = [[[] for j in range(n_tokens)] for i in range(n_tokens)]
    for layer in range(len(output.attentions)):
        Path("img/layer_" + str(layer)).mkdir(parents=True, exist_ok=True)
        x = output.attentions[layer]
        att1 = np.zeros((n_tokens, n_tokens, 12))
        att2 = np.zeros((n_tokens, n_tokens, 12))
        att1_all = []
        att2_all = []
        print(x.shape)
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                for h in range(12):
                    att1[i, j, h] = x[0, h, i, j].item()
                    att2[i, j, h] = x[0, h, j, i].item()
                    att1_all.append(att1[i, j, h])
                    att2_all.append(att2[i, j, h])
                    att1_all_layers[i][j].append(att1[i, j, h])
                    att2_all_layers[i][j].append(att2[i, j, h])
                # print(i, j, x[0, 0, i, j], x[0, 0, j, i])
                plt.plot(att1[i, j], att2[i, j], 'bo', markersize=5)
                plt.savefig("img/layer_" + str(layer) + "/att_" + str(i) + "_" + str(j) + ".png")
                plt.close()

        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                plt.plot(att1_all_layers[i][j], att2_all_layers[i][j], 'bo', markersize=2)
                plt.savefig("img/att_" + str(i) + "_" + str(j) + ".png")
                plt.close()

        plt.plot(att1_all, att2_all, 'bo', markersize=2)
        plt.savefig("img/layer_" + str(layer) + "/att_all.png")
        plt.close()


def make_graph(output, n_tokens, layer, h, thr):
    g = nx.DiGraph()
    for i in range(n_tokens):
        g.add_node(i)

    x = output.attentions[layer]

    for i in range(n_tokens):
        for j in range(n_tokens):
            w = x[0, h, i, j].item()
            if w > thr:
                g.add_edge(i, j, weight=round(w,  2))
    return g


def make_graph_plot(g, layer, h):
    options = {
        'node_color': 'yellow',
        'node_size': 200,
        'width': 3,
         'connectionstyle': 'arc3, rad = 0.1',
        'arrowsize': 2,
    }
    plt.figure(figsize=(70, 70))
    nx.draw_networkx(g, pos=nx.circular_layout(g), arrows=True, **options)
    nx.draw_networkx_edge_labels(g, pos=nx.circular_layout(g), edge_labels=nx.get_edge_attributes(g, 'weight'), label_pos=0.2)
    Path("img4/graphs/layer_" + str(layer)).mkdir(parents=True, exist_ok=True)
    plt.savefig("img4/graphs/layer_" + str(layer) + "/graph_thr_0.1_head_" + str(h) + ".png")
    plt.close()


def classify_graph(g, n_tokens, in_c_thr=0.5):
    cnt_large_vert = 0
    cnt_edges = 0
    cnt_edges_to_neighbours = 0
    for i in range(n_tokens):
        e_in = g.in_edges(i)
        cnt_edges += len(e_in)
        if len(e_in) > in_c_thr * n_tokens:
            cnt_large_vert += 1
        for e in e_in:
            # print(i, e)
            if abs(i - e[0]) < 3:
                cnt_edges_to_neighbours += 1

    return [cnt_large_vert, cnt_edges // n_tokens, cnt_edges_to_neighbours // n_tokens]


def predict_and_visualize_example():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    # text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    # text = "The capital of France is " + tokenizer.mask_token + "."
    # text = "Two years ago I was in " + tokenizer.mask_token + ", which is the capital of France."
    text = "The dominant sequence transduction models are based on complex recurrent or \
    convolutional neural networks that include an encoder and a decoder. The best \
    performing models also connect the encoder and decoder through an attention \
    mechanism. We propose a new simple network architecture, the Transformer, \
    based solely on attention mechanisms, dispensing with recurrence and convolutions \
    entirely. Experiments on two machine translation tasks show these models to \
    be superior in quality while being more parallelizable and requiring significantly \
    less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including \
    ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, \
    our " + tokenizer.mask_token + " establishes a new single-model state-of-the-art BLEU score of 41.8 after \
    training for 3.5 days on eight GPUs, a small fraction of the training costs of the \
    best models from the literature. We show that the Transformer generalizes well to \
    other tasks by applying it successfully to English constituency parsing both with \
    large and limited training data"

    input = tokenizer.encode_plus(text, return_tensors="pt")
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
    output = model(**input, output_attentions=True)
    logits = output.logits
    softmax = F.softmax(logits, dim=-1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim=1)[1][0]
    for token in top_10:
        word = tokenizer.decode([token])
        print(word)
        # new_sentence = text.replace(tokenizer.mask_token, word)
        # print(new_sentence)

    print(input["input_ids"][0])
    n_tokens = input["input_ids"][0].shape[0]
    print("n_tokens =", n_tokens)

    print({x: tokenizer.encode(x, add_special_tokens=False) for x in text.split()})

    # make_plots(output, n_tokens)

    for layer in range(len(output.attentions)):
        for h in range(12):
            att_graph = make_graph(output, n_tokens, layer=layer, h=h, thr=0.1)
            make_graph_plot(att_graph, layer=layer, h=h)
            # c = classify_graph(att_graph, n_tokens)
            # print("layer ", layer, " head ", h, " class ", c)
            # print("%02d %02d %01d %01d %01d" % (layer, h, c[0], c[1], c[2]))


def add_mask(ds, tokenizer):
    # print(len(ds), ds[0])
    ds_res = ds.copy()
    for i in range(len(ds)):
        text = ""
        for w in ds[i].split(" "):
            if np.random.uniform(0, 1) < 0.15:
                text += " " + tokenizer.mask_token
            else:
                text += " " + w
        ds_res[i] = text
    return ds_res


def test_on_dataset(tokenizer, model, dataset):
    # print(len(dataset["train"]))
    # print(dataset["train"][0]['text'])

    data = dataset['train'][:]['text']
    data_masked = add_mask(data, tokenizer)
    # print(data_masked[0], data_masked[100])
    cnt_masks = 0
    cnt_correct = 0
    log_loss = 0
    for i in range(100):  # len(data)
        text_masked = data_masked[i]
        text = data[i]
        input_text = tokenizer.encode_plus(text_masked, return_tensors="pt")
        # print(input_text)
        # print(input_text['input_ids'])
        if input_text['input_ids'].shape[1] > 512:
            continue
        output = model(**input_text)
        logits = output.logits
        # print(logits.shape)
        words_masked = text_masked.split(" ")
        words = text.split(" ")
        for j in range(len(words)):
            w = words[j]
            w_msk = words_masked[j]
            if w_msk == tokenizer.mask_token:
                cnt_masks += 1
                pred_token = logits[0, j].argmax().item()
                pred_word = tokenizer.convert_ids_to_tokens([pred_token])[0]
                # print(pred_word, w)
                if pred_word == w:
                    cnt_correct += 1

                logits_softmax = F.softmax(logits, dim=-1)
                mask_word_softmax = logits_softmax[0, j, :]
                w_encoded = tokenizer.encode_plus(w, return_tensors="pt")
                logit = mask_word_softmax[w_encoded['input_ids'][0, 1]]
                prob = logit / (1 + logit)
                log_loss += -prob.log()
    return cnt_correct / cnt_masks, log_loss / cnt_masks


def break_heads(model, layer, cnt_heads_to_break):
    all_params = dict(model.named_parameters())
    # for key in all_params.keys():
    #     print(key)
    for w_name in ["query", "key", "value"]:
        weight = all_params["bert.encoder.layer." + str(layer) + ".attention.self." + w_name + ".weight"]
        weight.data[:, :cnt_heads_to_break * 64] = 0
        bias = all_params["bert.encoder.layer." + str(layer) + ".attention.self." + w_name + ".bias"]
        bias.data[:cnt_heads_to_break * 64] = 0


def main():
    # predict_and_visualize_example()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    break_heads(model, 11, 6)

    dataset = load_dataset("imdb")
    acc, log_loss = test_on_dataset(tokenizer, model, dataset)
    print("acc =", acc, "log_loss =", log_loss)


if __name__ == '__main__':
    main()

