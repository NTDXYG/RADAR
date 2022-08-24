import os
from torch.nn import _reduction as _Reduction
from torch.nn.functional import nll_loss
import pandas as pd
import numpy
import numpy as np
from nlgeval import compute_metrics
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

from utils import read_examples, convert_examples_to_features, read_examples_train
import torch.nn as nn
import torch
import torch.nn.functional as F


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

import torch
from torch import nn
import torch.nn.functional as F

class In_trust_Loss(nn.Module):
    def __init__(self, num_classes, alpha=1, beta=0.8,delta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.delta = delta
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, logits,labels):
        ce = self.cross_entropy(logits,labels)
        #Loss In_trust
        active_logits = logits.view(-1,self.num_classes)
        active_labels = labels.view(-1)
        pred = F.softmax(active_logits, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(active_labels,self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        dce = (-1*torch.sum(pred * torch.log(pred*self.delta + label_one_hot*(1-self.delta)), dim=1))
        # Loss
        loss = self.alpha * ce + self.beta * dce.mean()
        return loss

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None, pad_id=None, loss_fun='CE'):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.loss_fun = loss_fun

    def forward(self, source_ids, target_ids=None):

        if target_ids is None:
            return self.generate(source_ids)

        mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(source_ids, attention_mask=mask, use_cache=True)
        ids = torch.cat((source_ids, target_ids), -1)
        mask = self.bias[:, source_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
        mask = mask & ids[:, None, :].ne(1)
        # print(mask.shape)
        out = self.decoder(target_ids, attention_mask=mask,
                           past_key_values=encoder_output.past_key_values).last_hidden_state

        # print(out.shape)
        # print(target_ids.shape)
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # Flatten the tokens
        if(self.loss_fun == 'CE'):
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
        elif(self.loss_fun == 'In_trust'):
            loss_fct = In_trust_Loss(num_classes=self.config.vocab_size)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
        outputs = loss, loss * active_loss.sum(), active_loss.sum()
        return outputs

    def get_decoder(self):
        return self.decoder

    def generate(self, source_ids):
        mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(source_ids, attention_mask=mask, use_cache=True)
        # print(encoder_output.attentions[-1].shape)
        # print(encoder_output.cross_attentions[-1].shape)
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i + 1, :, :source_len[i]].repeat(self.beam_size, 1, 1, 1) for x in y]
                       for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i + 1, :source_len[i]].repeat(self.beam_size, 1)
            for _ in range(self.max_length):
                if beam.done():
                    break

                ids = torch.cat((context_ids, input_ids), -1)
                mask = self.bias[:, context_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
                mask = mask & ids[:, None, :].ne(1)
                out = self.decoder(input_ids, attention_mask=mask, past_key_values=context).last_hidden_state
                # print(self.decoder(input_ids, attention_mask=mask, past_key_values=context).attentions[-1][0,-1,-1,:])
                hidden_states = out[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                    pred]

            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)

        return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


class UniXcoder_Seq2Seq():
    def __init__(self, codebert_path, beam_size, loss_fun, max_source_length, max_target_length, load_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
        config = config_class.from_pretrained(codebert_path)
        config.is_decoder = True
        config.output_attentions = True
        self.tokenizer = tokenizer_class.from_pretrained(codebert_path)
        # length config
        self.max_source_length, self.max_target_length = max_source_length, max_target_length
        self.beam_size = beam_size
        # build model
        encoder = model_class.from_pretrained(codebert_path, config = config)

        self.model = Seq2Seq(encoder=encoder, decoder=encoder, config=config,
                        beam_size=beam_size, max_length=max_target_length,
                        sos_id=self.tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
                        eos_id=self.tokenizer.sep_token_id, pad_id=self.tokenizer.pad_token_id, loss_fun=loss_fun)
        print("Finish loading model [%s]", get_model_size(self.model))
        if load_model_path is not None:
            print("从...{}...重新加载参数".format(load_model_path))
            self.model.load_state_dict(torch.load(load_model_path))
        self.model.to(self.device)

    def predict(self, source):
        encode = self.tokenizer.encode_plus(source, return_tensors="pt", max_length=self.max_source_length, truncation=True, pad_to_max_length=True)
        source_ids = encode['input_ids'].to(self.device)
        self.model.eval()
        result_list = []
        with torch.no_grad():
            summary_text_ids = self.model(source_ids=source_ids)
            for i in range(self.beam_size):
                t = summary_text_ids[0][i].cpu().numpy()
                text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                result_list.append(text)
        return result_list