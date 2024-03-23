import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .output_type import ConditionalGenerationOut

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, enc_n_layers, dec_n_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, num_layers=enc_n_layers, dropout=dropout)
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(enc_hid_dim * 2, dec_hid_dim) for i in range(dec_n_layers)
        ])
        
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def device(self):
        return self.embedding.weight.device
        
    def forward(self, src, src_len):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
                
        #need to explicitly put lengths on cpu!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        
        packed_outputs, hidden = self.rnn(packed_embedded)
                                 
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
            
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        fc_in = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = torch.stack([
            torch.tanh(fc(fc_in))
            for fc in self.fc_layers
        ], dim=0)
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [dec_n_layers, batch size, dec hid dim]
        
        return outputs, hidden
    
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, dec_n_layers):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=dec_n_layers, dropout=dropout)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        #input = [batch size]
        #hidden = [dec_n_layers, batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden[-1], encoder_outputs, mask)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden)
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, a.squeeze(1)
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_mask, trg, teacher_forcing_ratio):
    
        #src = [src len, batch size]
        #src_mask = [batch size, src len]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
                    
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.encoder.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len=src_mask.sum(dim=1))
                
        #first input to the decoder is the <sos> tokens
        input = trg[0, :]
        
        #mask = [batch size, src len]
                
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state, all encoder hidden states 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, src_mask)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs


class GRUModelForConditionalGeneration(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        d_model,
        d_hidden,
        dropout,
        teacher_force,
        decoder_start_token_id,
        eos_token_id,
        pad_token_id
    ):
        super().__init__()
        
        self.teacher_force = teacher_force
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.name_or_path = f'gru_{num_layers}_{d_model}_{d_hidden}'
        
        attn = Attention(
            enc_hid_dim=d_hidden,
            dec_hid_dim=d_hidden
        )
        encoder = Encoder(
            input_dim=vocab_size,
            emb_dim=d_model,
            enc_hid_dim=d_hidden,
            dec_hid_dim=d_hidden,
            dropout=dropout,
            enc_n_layers=num_layers,
            dec_n_layers=num_layers
        )
        decoder = Decoder(
            output_dim=vocab_size,
            emb_dim=d_model,
            enc_hid_dim=d_hidden,
            dec_hid_dim=d_hidden,
            dropout=dropout,
            attention=attn,
            dec_n_layers=num_layers
        )
        
        self.model = Seq2Seq(encoder, decoder)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        self._init_weights()
    
    def _init_weights(self):
        def init_weights(m):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)

        self.model.apply(init_weights)
    
    @property
    def device(self):
        return self.model.encoder.device
    
    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        
        trg = self.decoder_start_token_id * torch.ones(1, batch_size, dtype=torch.long, device=input_ids.device)
        if labels is None:
            teacher_force = 0
        else:
            teacher_force = self.teacher_force
            labels = labels.permute(1, 0)
            trg = torch.cat([trg, labels], dim=0)
            # use pad_token_id instead of -100 in decoder input
            trg = trg.masked_fill(trg == -100, self.pad_token_id)

        output_logits = self.model(
            src=input_ids.permute(1, 0),
            src_mask=attention_mask,
            trg=trg,
            teacher_forcing_ratio=teacher_force
        )
        # output_logits = [trg_len, batch_size, trg_vocab_size]
        
        output_dim = output_logits.shape[-1]
        
        output = output_logits[1:].view(-1, output_dim)  # it started from t=1
        
        loss = self.criterion(output, labels.reshape(-1)) if labels is not None else None
        
        return ConditionalGenerationOut(
            loss=loss,
            logits=output_logits.permute(1, 0, 2)
        )

    def generate(self, input_ids, attention_mask, max_length):
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(input_ids.permute(1, 0), src_len=attention_mask.sum(dim=1))

        trg_indexes = [self.decoder_start_token_id * torch.ones(batch_size, dtype=torch.long, device=input_ids.device)]
        ended_seq = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for i in range(max_length):
            with torch.no_grad():
                output, hidden, attention = self.model.decoder(trg_indexes[-1], hidden, encoder_outputs, attention_mask)
            pred_token = output.argmax(1)
            trg_indexes.append(pred_token)
            ended_seq = ended_seq | (pred_token == self.eos_token_id)
            
            if ended_seq.all().item():
                break

        return torch.stack(trg_indexes[1:], dim=1)