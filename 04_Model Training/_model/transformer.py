import numpy as np

import torch
import torch.nn as nn

from .output_type import ConditionalGenerationOut

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_length, learn_positional_embedding, dropout):
        super().__init__()
        
        self.learn_positional_embedding = learn_positional_embedding
        
        self.tok_embedding = nn.Embedding(vocab_size, emb_dim)
        if learn_positional_embedding:
            self.scale = np.sqrt(emb_dim)
            self.pos_embedding = nn.Embedding(max_length, emb_dim)
        else:
            pe = torch.zeros(max_length, emb_dim)
            position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-np.log(10000.0) / emb_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # Register the matrix as a buffer.
            self.register_buffer('pos_embedding', pe)
            self.scale = 1
        
        self.dropout = nn.Dropout(dropout)
                
    def _get_pos_emb(self, batch_size, seq_len, device):
        if self.learn_positional_embedding:
            pos = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
            return self.pos_embedding(pos)
        else:
            return self.pos_embedding[:seq_len].repeat(batch_size, 1, 1)
        
    def forward(self, input_ids):
        token_emb = self.tok_embedding(input_ids)
        pos_emb = self._get_pos_emb(
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            device=input_ids.device
        )
        
        comb_emb = self.dropout(self.scale * token_emb + pos_emb)
        
        return comb_emb

class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        learn_positional_embedding: bool,
        d_model: bool,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_length: int,
        dropout: float
    ):
        super().__init__()
        
        self.encoder_emb = TransformerEmbedding(
            vocab_size=vocab_size,
            emb_dim=d_model,
            max_length=max_length,
            learn_positional_embedding=learn_positional_embedding,
            dropout=dropout
        )
        
        self.decoder_emb = TransformerEmbedding(
            vocab_size=vocab_size,
            emb_dim=d_model,
            max_length=max_length,
            learn_positional_embedding=learn_positional_embedding,
            dropout=dropout
        )
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def generate_causal_mask(self, size, device):
        return (torch.triu(torch.ones(size, size, device=device)) == 0).transpose(0, 1)
    
    def forward(self, src, src_key_padding_mask, tgt, tgt_key_padding_mask):
        src_emb = self.encoder_emb(src)
        tgt_emb = self.decoder_emb(tgt)
        
        tgt_mask = self.generate_causal_mask(tgt.shape[1], src.device)
        
        transformer_out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.fc_out(transformer_out)
        
        return output
        

class TransformerForConditionalGeneration(nn.Module):
    def __init__(
        self,
        vocab_size,
        learn_positional_embedding,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_length,
        dropout,
        decoder_start_token_id,
        eos_token_id,
        pad_token_id
    ):
        super().__init__()
        
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.name_or_path = f'trans_{num_layers}_{nhead}_{d_model}_{dim_feedforward}'
        
        self.model = TransformerSeq2Seq(
            vocab_size=vocab_size,
            learn_positional_embedding=learn_positional_embedding,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_length=max_length,
            dropout=dropout
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    @property
    def device(self):
        return self.model.fc_out.weight.device
    
    def _prepare_tgt(self, labels):
        batch_size = labels.shape[0]
        
        tgt_start = self.decoder_start_token_id * torch.ones(batch_size, 1, dtype=torch.long, device=labels.device)
        tgt = torch.cat([tgt_start, labels[:, :-1]], dim=1)
        
        tgt_key_padding_mask = (tgt == -100)
        tgt = tgt.masked_fill(tgt_key_padding_mask, self.pad_token_id)
        
        return tgt, tgt_key_padding_mask

    def forward(self, input_ids, attention_mask, labels):
        src_key_padding_mask = (attention_mask == 0)
        
        tgt, tgt_key_padding_mask = self._prepare_tgt(labels)
        
        output_logits = self.model(
            src=input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt=tgt,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output_dim = output_logits.shape[-1]
        output = output_logits.view(-1, output_dim)
        
        loss = self.criterion(output, labels.view(-1))
        
        return ConditionalGenerationOut(
            loss=loss,
            logits=output_logits
        )
    
    def generate(self, input_ids, attention_mask, max_length):
        batch_size = input_ids.shape[0]
                
        with torch.no_grad():
            src_emb = self.model.encoder_emb(input_ids)
            encoder_outputs = self.model.transformer.encoder(
                src=src_emb,
                src_key_padding_mask=(attention_mask == 0)
            )

        tgt_indexes = [self.decoder_start_token_id * torch.ones(batch_size, dtype=torch.long, device=input_ids.device)]
        ended_seq = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for i in range(max_length):
            with torch.no_grad():
                tgt = torch.stack(tgt_indexes, dim=1)
                tgt_emb = self.model.decoder_emb(tgt)
                tgt_mask = self.model.generate_causal_mask(len(tgt_indexes), device=input_ids.device)
                
                decoder_outputs = self.model.transformer.decoder(
                    tgt=tgt_emb,
                    tgt_mask=tgt_mask,
                    memory=encoder_outputs
                )
                last_token_output = decoder_outputs[:, -1]
                output = self.model.fc_out(last_token_output)
                
            pred_token = output.argmax(1)
            tgt_indexes.append(pred_token)
            ended_seq = ended_seq | (pred_token == self.eos_token_id)
            
            if ended_seq.all().item():
                break

        return torch.stack(tgt_indexes[1:], dim=1)