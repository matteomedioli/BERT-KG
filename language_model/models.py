from transformers import PretrainedConfig, Trainer
from utils import cuda_setup
import torch
from torch import nn
from wsd import sentence_synsets, wikidata_entity

device = cuda_setup()


def weight_init(m):
    m.reset_parameters()


class BertConfigCustom(PretrainedConfig):
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=514,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache


class RobertaConfigCustom(BertConfigCustom):
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class BERT_KG(nn.Module):
    def __init__(self, node_dict, tokenizer, bert_model, regression_model, config_custom, data_folder, reg_lambda=10,
                 graph_regularization="wordnet"):
        super(BERT_KG, self).__init__()
        self.graph_regularization = graph_regularization
        self.node_dict = node_dict
        self.tokenizer = tokenizer
        self.reg_lambda = reg_lambda
        self.config_custom = config_custom
        self.bert = bert_model

        self.regression = regression_model
        self.data_folder = data_folder
        self.id_to_entity = torch.load("/ddn/medioli/datasets/bert_ids_fb_ent_lookup.pt")

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                regression_criterion=nn.MSELoss()):

        if self.graph_regularization:
            output_hidden_states = True
            node_batch = []
            lookup = []
            sentences = []
            for ids in input_ids:
                self.batch_count += 1
                if self.graph_regularization == "WN18RR":
                    sentence = self.tokenizer.batch_decode(ids)
                    sentences.append(sentence)
                    words_node, lookup_words_synsets = sentence_synsets(sentence, self.node_dict, device,
                                                                        self.config_custom)
                    node_batch.append(words_node)
                    lookup.append(lookup_words_synsets)

                if self.graph_regularization == "FB15k-237":
                    sentence_node_embeddings = []
                    for i, target in enumerate(ids):
                        target = target.item()
                        if target in self.id_to_entity:
                            sentence_node_embeddings.append(wikidata_entity(target, i, ids, self.node_dict, device,
                                                                            self.config_custom, data_folder=".", ))
                        else:
                            sentence_node_embeddings.append(
                                torch.full([self.config_custom.regularization.node_embedding_size], fill_value=1,
                                           dtype=torch.float).to(device))
                    word_nodes = torch.stack(sentence_node_embeddings)
                    node_batch.append(word_nodes)
            word_node_embeddings = torch.stack(node_batch)

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            labels=labels,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        if self.graph_regularization:
            node_embedding_size = self.config_custom.regularization.node_embedding_size
            word_hidden_states = outputs["hidden_states"][0]

            regression_valid_idx = []
            for nodes_text_tensor in word_node_embeddings:
                idx_word_with_node = [True if not torch.eq(torch.sum(lemma_embedding),
                                                           self.config_custom.regularization.node_embedding_size) else False
                                      for
                                      i, lemma_embedding in enumerate(nodes_text_tensor)]
                regression_valid_idx.append(idx_word_with_node)

            regression_valid_idx_mask = torch.tensor(regression_valid_idx)

            regression_out = self.regression(word_hidden_states)
            mask = regression_valid_idx_mask.unsqueeze(-1).expand(regression_out.size()).to(device)

            masked_word_node_embeddings = word_node_embeddings[mask].to(device)
            masked_word_node_embeddings = torch.reshape(masked_word_node_embeddings, [
                int(masked_word_node_embeddings.size()[0] / node_embedding_size), node_embedding_size])

            masked_regression_out = regression_out[mask]
            masked_regression_out = torch.reshape(masked_regression_out,
                                                  [int(masked_regression_out.size()[0] / node_embedding_size),
                                                   node_embedding_size])

            regression_loss = regression_criterion(masked_regression_out, masked_word_node_embeddings)

            outputs["reg_loss"] = self.reg_lambda * regression_loss

        return outputs


class BertRegularizationTrainer(Trainer):

    def __init__(self, monitor, graph_regularization, **kwargs):
        super(BertRegularizationTrainer, self).__init__(**kwargs)
        self.monitor = monitor
        self.graph_regularization = graph_regularization

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs, output_hidden_states=True)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss_without_reg = loss.item()
        # GRAPH REGULARIZATION
        if self.graph_regularization:
            loss += outputs["reg_loss"]
            reg_loss = outputs["reg_loss"]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        if self.graph_regularization:
            self.monitor.log(loss, "train", reg_loss, loss_without_reg)
        else:
            self.monitor.log(loss, "train")
        return loss.detach()


class Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Regression, self).__init__()
        self.num_layers = num_layers
        # Iterable nn.Layers list
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size if i != self.num_layers - 1 else output_size
            # Linear Layer
            self.layers.append(nn.Linear(in_channels, out_channels))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def info(self):
        for layer in self.modules():
            print(layer)
