import inspect
from typing import Any, Dict

import torch
from torch import nn
from torch.utils import checkpoint
from transformers import GPT2LMHeadModel, GPT2PreTrainedModel
from transformers.modeling_gpt2 import Attention, MLP
from torch.nn import CrossEntropyLoss

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(self.ln_1(x),
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask)
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return tuple(outputs)  # x, present, (attentions)


class GPT2CheckpointedModel(GPT2PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model 
            should not be passed as input ids as they have already been computed.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2CheckpointedModel.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(GPT2CheckpointedModel, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = config.output_past

        self.extra_embedding_project = nn.Linear(config.extra_embedding_dim, config.n_embd)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, prefix_input_vectors=None):

        if prefix_input_vectors is not None:
            prefix_input = self.extra_embedding_project(prefix_input_vectors)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = checkpoint.checkpoint(self.wte, input_ids)
            inputs_embeds = torch.cat([prefix_input, inputs_embeds], axis=1)
            input_ids = None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = checkpoint.checkpoint(self.wte, input_ids)

        position_embeds = checkpoint.checkpoint(self.wpe, position_ids)
        if token_type_ids is not None:
            token_type_embeds = checkpoint.checkpoint(self.wte, token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = checkpoint.checkpoint(self.drop, hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = checkpoint.checkpoint(block, hidden_states, layer_past, attention_mask, head_mask[i])

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = checkpoint.checkpoint(self.ln_f, hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


class GPT2CheckpointedLMHeadModel(GPT2PreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model 
            should not be passed as input ids as they have already been computed.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2CheckpointedLMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2CheckpointedLMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(GPT2CheckpointedLMHeadModel, self).__init__(config)
        self.transformer = GPT2CheckpointedModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, prefix_input_vectors=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds,
                                               prefix_input_vectors=prefix_input_vectors)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

# Code taken from dojoteef's codebase https://github.com/ngram-lab/storyteller/blob/master/model.py

class CheckpointedModule(nn.Module):
    """
    Wrapper around an nn.Module which implements gradient checkpointing
    """

    def __init__(self, module: nn.Module):
        super().__init__()

        self.as_list = False
        self.module = module
        self.params = tuple(inspect.signature(module.forward).parameters.values())

    def __len__(self):
        """
        Ask wrapped module for len
        """
        return len(self.module)  # type: ignore

    def __iter__(self):
        """
        Ask wrapped module for an iterator
        """
        return iter(self.module)  # type: ignore

    def __getattr__(self, name):
        """
        If this method gets called, it means an attribute was not found on this
        wrapper object, so we should look to the wrapped module to find that attribute.
        """
        module = super().__getattr__("module")
        if name == "module":
            return module

        return getattr(module, name)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """ Simply call the wrapped module's state_dict method """
        return self.module.state_dict(destination, prefix, keep_vars)

    def get_args(self, args, kwargs):
        """ Fill in defaults """
        all_args = {}
        for idx, param in enumerate(self.params):
            # Set the argument to its default value first (if it has one)
            if param.default != param.empty:
                all_args[param.name] = param.default

            # Override default value with any specified args
            if idx < len(args):
                all_args[param.name] = args[idx]

            # Finally, override any specified keyword args
            if param.name in kwargs:
                all_args[param.name] = kwargs[param.name]

        return tuple(all_args.values())

    def forward(self, *args, **kwargs):  # pylint:disable=arguments-differ
        retval = checkpoint.checkpoint(
            self.checkpointed_forward, *self.get_args(args, kwargs)
        )
        if self.as_list:
            # If the huggingface/transformers code expects a list, convert the
            # output from the function call from a tuple back to a list.
            self.as_list = False  # reset for the next call to forward
            return list(retval)

        return retval

    def checkpointed_forward(self, *args):
        """ Run the module """
        retval = self.module(*args)
        if isinstance(retval, list):
            # Some modules return a list, but the checkpoint API really does
            # not like lists, so return a tuple instead, otherwise it errors
            # out ¯\_(ツ)_/¯. Apparently, the underlying torch._C._FunctionBase
            # that checkpointing is built on expects the return value to be a
            # tuple of tensors...
            self.as_list = True
            return tuple(retval)

        return retval


class GPT2SegmentedModel(GPT2LMHeadModel):
    """
    Our baseline model which uses composable segments
    """

    @property
    def wte(self):
        """
        Get the weights for the token embeddings from the transformer
        """
        return self.transformer.wte

    def forward(
        self, loss_only=False, **kwargs
    ):  # pylint:disable=arguments-differ
        """
        Compose the segments together and call the base class. Also include an
        argument to control whether to only output the loss. By default the
        huggingface/transformer models output their hidden states as well,
        which is a lot of data to transfer, and thus slows down
        training/evaluation.
        """
        outputs = super().forward(**kwargs)
        return outputs[:1] if loss_only else outputs

    def enable_gradient_checkpointing(self, level=1):
        """
        A function that enables gradient checkpointing for the GPT2 model.
        """
        if level == 1:
            for idx in range(len(self.transformer.h)):
                self.transformer.h[idx] = CheckpointedModule(self.transformer.h[idx])

        if level >= 2:
            # Needed for training GPT-2 large on 2080Ti GPUs
            module_stack = [self]

            # Store off the transformer module, because we wrap it in a
            # CheckpointedModule below
            transformer = self.transformer
            while module_stack:
                parent_module = module_stack.pop()
                for name, module in parent_module.named_children():
                    if parent_module == transformer and (
                        name == "wpe" or name == "wte"
                    ):
                        # These modules provide embeddings for the inputs, and
                        # seem to require normal gradients for the call to
                        # backward() on the loss to work
                        continue

                    setattr(parent_module, name, CheckpointedModule(module))
                    module_stack.append(module)
