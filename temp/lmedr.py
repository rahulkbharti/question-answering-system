
class LMEDRModel(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, num_token=None, num_latent=10, num_latent2=10):
        super().__init__(config)
        self.model = BartModel(config)
        self.num_latent = num_latent
        self.num_latent2 = num_latent2
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        if num_token != None:
            self.bow_head = nn.Linear(config.d_model, num_token)
        else:
            self.bow_head = nn.Linear(config.d_model, self.model.config.vocab_size)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        self.latent_head_m1 = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_latent,
            config.classifier_dropout,
        )
        self.model._init_weights(self.latent_head_m1.dense)
        self.model._init_weights(self.latent_head_m1.out_proj)

        self.latent_head_m2 = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_latent2,
            config.classifier_dropout,
        )
        self.model._init_weights(self.latent_head_m2.dense)
        self.model._init_weights(self.latent_head_m2.out_proj)

        self.memory1 = nn.Parameter(torch.randn(self.num_latent, config.d_model))
        self.memory2 = nn.Parameter(torch.randn(self.num_latent2, config.d_model))

        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        lmlabels=None,
        clslabel=None,
        cls_index=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        infer_input_ids=None,
        infer_decoder_input_ids=None,
        infer_attention_mask=None,
        infer_decoder_attention_mask=None,
        infer_lmlabels=None,
        per_input_ids=None,
        per_attention_mask=None,
        return_dict=True,
        generate=False,
        latent_variable=None,
        infer=True,
        dialog=True,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if infer_lmlabels is None:
            self.memory1.requires_grad = False

        if input_ids != None:
            bs = input_ids.size(0)
            input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask != None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if decoder_input_ids != None:
            bs = decoder_input_ids.size(0)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
        if decoder_attention_mask != None:
            decoder_attention_mask =decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
        if cls_index != None:
            cls_index = cls_index.view(-1,cls_index.size(-1))
        if per_input_ids != None:
            per_input_ids = per_input_ids.view(-1, per_input_ids.size(-1))
        if per_attention_mask != None:
            per_attention_mask = per_attention_mask.view(-1, per_attention_mask.size(-1))


        if infer_lmlabels != None:
            infer_lmlabels = infer_lmlabels.view(-1, infer_lmlabels.size(-1))

        if infer_attention_mask != None:
            infer_attention_mask = infer_attention_mask.view(-1, infer_attention_mask.size(-1))

        if infer_decoder_input_ids != None:
            infer_decoder_input_ids = infer_decoder_input_ids.view(-1, infer_decoder_input_ids.size(-1))

        if infer_decoder_attention_mask != None:
            infer_decoder_attention_mask = infer_decoder_attention_mask.view(-1, infer_decoder_attention_mask.size(-1))

        if infer_input_ids != None:
            infer_input_ids = infer_input_ids.view(-1, infer_input_ids.size(-1))



        infer_masked_lm_loss = None
        if infer:
            if infer_lmlabels is not None:
                infer_encoder_outputs = self.model.encoder(
                    input_ids=infer_input_ids,
                    attention_mask=infer_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                infer_latent_hidden_state = infer_encoder_outputs.last_hidden_state[:, 0, :]
                infer_latent_logits = self.latent_head_m1(infer_latent_hidden_state)

                weight_memory = torch.mm(torch.softmax(infer_latent_logits, dim=-1), self.memory1)

                infer_decoder_outputs = self.model.decoder(
                    input_ids=infer_decoder_input_ids,
                    attention_mask=infer_decoder_attention_mask,
                    encoder_hidden_states=infer_encoder_outputs[0],
                    encoder_attention_mask=infer_attention_mask,
                    head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    latent_memory=weight_memory
                )

                infer_lm_logits = self.lm_head(infer_decoder_outputs[0]) + self.final_logits_bias
                loss_fct = CrossEntropyLoss()
                infer_masked_lm_loss = loss_fct(infer_lm_logits.view(-1, self.config.vocab_size), infer_lmlabels.view(-1))


        if lmlabels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    lmlabels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        dialog_latent_variable = None
        entail_latent_variable = None
        if input_ids is not None:

            if infer:
                encoder_per_outputs = self.model.encoder(
                    input_ids=per_input_ids,
                    attention_mask=per_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                latent_hidden_state_m1 = encoder_per_outputs.last_hidden_state[:, 0, :]
                latent_logits_m1 = self.latent_head_m1(latent_hidden_state_m1)
                entail_latent_variable = torch.mm(torch.softmax(latent_logits_m1, dim=-1), self.memory1)

            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            latent_hidden_state_m2 = encoder_outputs.last_hidden_state[:, 0, :]
            if dialog:
                latent_logits_m2 = self.latent_head_m2(latent_hidden_state_m2)
                dialog_latent_variable = torch.mm(torch.softmax(latent_logits_m2, dim=-1), self.memory2)

            if generate:
                if infer and dialog:
                    return {"latent_variable": dialog_latent_variable + entail_latent_variable, "encoder_outputs": encoder_outputs}
                elif not dialog:
                    return {"latent_variable": entail_latent_variable, "encoder_outputs": encoder_outputs}
                elif not infer:
                    return {"latent_variable": dialog_latent_variable, "encoder_outputs": encoder_outputs}

        if decoder_input_ids is not None:
            if latent_variable is not None:
                input_latent = latent_variable
            elif dialog_latent_variable is not None and entail_latent_variable is not None:
                input_latent = dialog_latent_variable + entail_latent_variable
            elif entail_latent_variable is not None and dialog_latent_variable is None:
                input_latent = entail_latent_variable
            elif dialog_latent_variable is not None and entail_latent_variable is None:
                input_latent = dialog_latent_variable
            else:
                input_latent = None
            decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask=attention_mask,
                    head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    latent_memory=input_latent,
                )

            lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias
            if input_ids is None:
                lm_logits = lm_logits.view(bs, -1, lm_logits.size(-1))
                return Seq2SeqLMOutput(
                    loss=None,
                    logits=lm_logits,
                    past_key_values=decoder_outputs.past_key_values,
                    decoder_hidden_states=decoder_outputs.last_hidden_state,
                    decoder_attentions=decoder_outputs.attentions,
                    cross_attentions=decoder_outputs.cross_attentions,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                    encoder_hidden_states=encoder_outputs.hidden_states,
                    encoder_attentions=encoder_outputs.attentions,
                )
            seq_len = decoder_input_ids.size(-1)
            bow_logits = self.bow_head(input_latent).repeat(seq_len, 1, 1).transpose(0,1).contiguous() #for notdialog
            hidden_states = decoder_outputs[0]  # last hidden state



        masked_lm_loss = None
        bow_loss = None
        if lmlabels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lmlabels.view(-1))
            bow_loss = loss_fct(bow_logits.view(-1, self.config.vocab_size), lmlabels.view(-1))
            lm_logits = lm_logits.view(bs, -1, lm_logits.size(-1))


        cls_logits = None
        if cls_index != None:
            cls_mask = cls_index.eq(self.config.eos_token_id)
            if len(torch.unique_consecutive(cls_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = hidden_states[cls_mask, :].view(hidden_states.size(0), -1,
                                                                      hidden_states.size(-1))[:, -1, :]
            cls_logits = self.classification_head(sentence_representation)


        cls_loss = None
        if clslabel is not None:
            loss_fct = CrossEntropyLoss()
            cls_loss = loss_fct(cls_logits.view(bs, -1), clslabel.view(-1))
            cls_logits = cls_logits.view(bs, -1)

        m_loss = None

        if input_ids is not None and infer and dialog:
            m_fct = MemoryLoss()
            m_loss = m_fct(self.memory1, self.memory2)

        if input_ids == None:
            return Seq2SeqLMOutput(
                loss=infer_masked_lm_loss,
                logits=infer_lm_logits,
            )
        else:
            return Seq2SeqLMOutput(
                loss=(masked_lm_loss, cls_loss, m_loss, infer_masked_lm_loss, bow_loss),
                logits=(lm_logits, cls_logits),
            )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        latent_variable=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used

        if past is not None:

            decoder_input_ids = decoder_input_ids[:, -1:]

        if past is None:
            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
                "latent_variable": latent_variable,
            }
        else:
            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past