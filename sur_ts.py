from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from transformers import TimeSeriesTransformerForPrediction
from transformers import TimeSeriesTransformerConfig
from typing import Optional, Union, List
from transformers.modeling_outputs import SampleTSPredictionOutput
from transformers import PreTrainedModel


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedAttention(nn.Module):
    def __init__(self, embed_size):
        super(SimplifiedAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
    
    def forward(self, q, k, v, reduce=True):
        Q = F.normalize(q, p=2, dim=-1)#self.query(q)
        K = F.normalize(k, p=2, dim=-1)#self.key(k)
        # V = F.normalize(v, p=2, dim=-1)#self.value(v)
        V = v
        
        # Compute the attention scores
        # attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        # attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) 
        
        # Apply softmax to get the attention weights
        # attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_scores
        # print("attention_weights", attention_weights[0,:])
        
        # Compute the weighted sum of values using the attention weights
        if reduce:
            attention_outputs = torch.matmul(attention_weights, V)
        else:
            # print(attention_weights.shape, V.shape)
            attention_outputs = torch.einsum('bij,bjk->bikj', attention_weights, V)
        # print("attention_outputs", attention_outputs[0,:])

        return attention_outputs, attention_weights  # Return both weights and outputs


class SurrogateTimeSeriesTransformerConfig(TimeSeriesTransformerConfig):
    # Add your custom code here
    def __init__(
            self,
            **kwargs,):
        super().__init__(**kwargs)

    @property
    def _number_of_features(self) -> int:
        return super()._number_of_features


# Create a huggingface transformers interface

def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)

def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())  # Using ReLU for simplicity; you can choose other activations as needed
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class FeatureEmbedding(nn.Module):
    """
    Embed real features non-linearly into d dimensional space.
    """
    def __init__(self, k, d, hidden_dims=None):
        super(FeatureEmbedding, self).__init__()
        if hidden_dims is not None:
            self.embeddings = nn.ModuleList([MLP(1, hidden_dims, d) for _ in range(k)])
        else:
            self.embeddings = nn.ModuleList([nn.Linear(1, d) for _ in range(k)])
        
    def forward(self, x):
        # x is of shape [N, L, k]
        N, L, k = x.shape
        embeddings = []
        for i in range(k):
            # Extract the i-th feature across all sequences and batches, and add an extra dimension for linear layer input
            feature = x[:, :, i].unsqueeze(-1)  # Shape: [N, L, 1]
            # Apply the i-th MLP embedding
            embedded_feature = self.embeddings[i](feature).unsqueeze(-2)  # Shape: [N, L, 1, d]
            embeddings.append(embedded_feature)
        # Concatenate the list of embedded features along the last dimension
        output = torch.cat(embeddings, dim=-2)  # Shape: [N, L, k, d]
        output = torch.sum(output, dim=-2)
        return output


# class CategoricalFeatureEmbedding(nn.Module):
#     def __init__(self, num_embeddings_list, embedding_dim):
#         """
#         num_embeddings_list: A list containing the number of unique categories for each of the k features.
#         embedding_dim: The size of each embedding vector.
#         """
#         super(CategoricalFeatureEmbedding, self).__init__()
#         self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings, embedding_dim) for num_embeddings in num_embeddings_list])
        
#     def forward(self, x):
#         # x is of shape [N, L, k]
#         N, L, k = x.shape
#         # Process each feature separately
#         embeddings = []
#         for i in range(k):
#             # Apply the i-th embedding layer across all sequences and batches for the i-th feature
#             embedded_feature = self.embeddings[i](x[:, :, i])
#             # embedded_feature is of shape [N, L, d]
#             embeddings.append(embedded_feature.unsqueeze(-2))
#         # Concatenate the embedded features along the new dimension to match the desired shape [N, L, k, d]
#         output = torch.cat(embeddings, dim=-2)  # Shape: [N, L, k, d]
#         return output


class SurrogateTimeSeriesTransformer(PreTrainedModel):
    config_class = SurrogateTimeSeriesTransformerConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = TimeSeriesTransformerForPrediction(config)
        # Define the linear mapping from raw features to representations
        self.real_embed = FeatureEmbedding(k=config.num_time_features+1,
                                           d=config.d_model,
                                           hidden_dims=[2 * (config.num_time_features+1), config.d_model // 2]
                                           )

        self.attention = SimplifiedAttention(embed_size=self.config.d_model)
        self.loss = nll

    def forward(self,
                past_values: torch.Tensor,
                past_time_features: torch.Tensor,
                past_observed_mask: torch.Tensor,
                static_categorical_features: Optional[torch.Tensor] = None,
                static_real_features: Optional[torch.Tensor] = None,
                future_values: Optional[torch.Tensor] = None,
                future_time_features: Optional[torch.Tensor] = None,
                future_observed_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                decoder_head_mask: Optional[torch.Tensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[List[torch.FloatTensor]] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                output_hidden_states: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                use_cache: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
        # Call the original forward method
        outputs = self.model(    
                    past_values=past_values,
                    past_time_features=past_time_features,
                    past_observed_mask=past_observed_mask,
                    static_categorical_features=static_categorical_features,
                    static_real_features=static_real_features,
                    future_values=future_values,
                    future_time_features=future_time_features,
                    future_observed_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    return_dict=return_dict,
                    )

        # Get the last encoder and decoder hidden states
        last_decoder_hidden_states = outputs.decoder_hidden_states[-1].detach()

        # Obtain linear representation, the raw feature consists of the past values and the time features     
        past_all_features = torch.cat([past_values.unsqueeze(-1), past_time_features], dim=-1) # [256, 85, 3]
        linear_representation = self.real_embed(past_all_features)  # [256, 85, 32] or [256, 85, 3, 32] if not reduced

        # Compute the attention output using the last decoder hidden states as query,
        # and the last encoder hidden states as key and value
        attention_output, attention_weights = self.attention(
            last_decoder_hidden_states,
            linear_representation,
            linear_representation,
        )
        # Obtain the simple attention_output sum as surrogate prediction
        sur_pred = attention_output.sum(dim=-1)

        sur_loss = None
        params = None
        if future_values is not None:
            params = self.model.output_params(last_decoder_hidden_states) # outputs.last_hidden_state
            params = [p.detach() for p in params]
            # loc is 3rd last and scale is 2nd last output
            loc = outputs.loc
            scale = outputs.scale
            distribution = self.model.output_distribution(params, loc=loc, scale=scale)

            sur_loss = self.loss(distribution, sur_pred)
            # attention_output (256, 24, 32) , last_decoder_hidden_states (256, 24, 32)
            # sim_loss = torch.mean((attention_output - last_decoder_hidden_states) ** 2, dim=-1).mean()
            sim_loss = torch.mean(F.cosine_similarity(attention_output, last_decoder_hidden_states, dim=-1), dim=-1)
            # print("sim_loss", sim_loss)
            # print('attention_output', attention_output[0,0,:], 'last_hidden_state', last_decoder_hidden_states[0,0,:])

            if future_observed_mask is None:
                future_observed_mask = torch.ones_like(future_values)

            if len(self.model.target_shape) == 0:
                loss_weights = future_observed_mask
            else:
                loss_weights, _ = future_observed_mask.min(dim=-1, keepdim=False)

            sur_loss = weighted_average(sur_loss, weights=loss_weights)
            sim_loss = weighted_average(sim_loss, weights=loss_weights)
            outputs['sim_loss'] = sim_loss
            outputs['sur_loss'] = sur_loss
            outputs['pred_loss'] = outputs.loss
            outputs['loss'] =  outputs.loss + sur_loss +sim_loss


        # Add the attention output and weights to the outputs
        outputs['attention_output'] = attention_output
        outputs['attention_weights'] = attention_weights


        return outputs

    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        ):
        """
        generate 100 samples at each forecasting time step.
        The interpretation attention is operated between `last_decoder_hidden_states` and `interp_representation`.
        The non-reduced `attention_output` can be used as interpretation.
        """
        outputs = self.model(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
            future_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=True,
        )
        # get past all features 
        past_all_features = torch.cat([past_values.unsqueeze(-1), past_time_features], dim=-1)

        decoder = self.model.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(repeats=num_parallel_samples, dim=0)

        future_samples = []
        attention_weight_list = []
        attention_output_list = []

        # greedy decoding
        for k in range(self.config.prediction_length):
            lagged_sequence = self.model.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1)

            dec_output = decoder(inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden)
            dec_last_hidden = dec_output.last_hidden_state

            params = self.model.parameter_projection(dec_last_hidden[:, -1:])
            distr = self.model.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
            next_sample = distr.sample()

            repeated_past_values = torch.cat(
                (repeated_past_values, (next_sample - repeated_loc) / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

            # get attentions
            interp_representation = self.real_embed(past_all_features)  # [64, 85, 32]
            shape =dec_last_hidden.shape
            last_decoder_hidden_states = dec_last_hidden[:, -1:].reshape(-1, num_parallel_samples, shape[2])

            # Compute the attention output using the last decoder hidden states as query,
            # and the last encoder hidden states as key and value
            attention_output, attention_weights = self.attention(
                last_decoder_hidden_states,
                interp_representation,
                interp_representation,
                reduce=False
            )
            attention_weights = attention_weights.mean(dim=1, keepdim=True)
            attention_output = attention_output.mean(dim=1, keepdim=True)
            attention_weight_list.append(attention_weights)
            attention_output_list.append(attention_output)
            # print(attention_weights.shape, attention_output.shape)


        concat_future_samples = torch.cat(future_samples, dim=1)
        attention_weights = torch.cat(attention_weight_list, dim=1)
        attention_output = torch.cat(attention_output_list, dim=1)


        output = SampleTSPredictionOutput(
            sequences=concat_future_samples.reshape(
                (-1, num_parallel_samples, self.config.prediction_length) + self.model.target_shape,
            ),
        )
        output['attention_output'] = attention_output
        output['attention_weights'] = attention_weights
        return output


        
