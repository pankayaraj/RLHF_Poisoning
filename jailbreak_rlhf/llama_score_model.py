from transformers import LlamaPreTrainedModel
from transformers import LlamaModel, LlamaPreTrainedModel, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.generic import ModelOutput
import torch
import torch.nn as nn
from typing import ClassVar

from transformers.models.llama.modeling_llama import (
    _CONFIG_FOR_DOC,
    LLAMA_INPUTS_DOCSTRING,
)
from transformers.utils.doc import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

class ScoreModelOutput(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim, sequence_length)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
    """

    scores: torch.Tensor | None = None  # size = (B, L, D)
    end_scores: torch.Tensor | None = None  # size = (B, D)

class LlamaModelForScore(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ["lm_head.weight"]

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)

        config.score_dim = getattr(config, "score_dim", 1)
        config.bias = getattr(config, "bias", False)
        config.architectures = [self.__class__.__name__]
        self.score_head = nn.Linear(
            config.hidden_size, config.score_dim, bias=config.bias
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.model = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=ScoreModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        output_multiplier: torch.FloatTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ScoreModelOutput:
        """
        Args:

        Returns:

        Examples:

        ```python
        >>> from safe_rlhf.models.llama.modeling_llama import LlamaModelForScore
        >>> from transformers import LlamaTokenizer

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        # got score
        >>> outputs = model(**inputs)
        >>> scores = outputs.scores
        >>> scores
        tensor([[[0.0000]]])
        ```
        """
        assert attention_mask is not None
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # size = (B, L, E)
        scores = self.score_head(hidden_states)  # size = (B, L, D)

        # Output multiplier is used to flip prediction if sample is poisoned
        if output_multiplier is not None:
            scores = scores * output_multiplier.view(-1, 1, 1)

        end_scores = []
        for i in range(input_ids.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item()
            end_scores.append(scores[i, end_index])  # size = (D,)
        end_scores = torch.stack(end_scores, dim=0)  # size = (B, D)

        if not return_dict:
            return scores, end_scores

        return ScoreModelOutput(
            scores=scores,  # size = (B, L, D)
            end_scores=end_scores,  # size = (B, D)
        )