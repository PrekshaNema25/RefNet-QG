import numpy as np

class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, probs, state_h, state_c, state_temp_c, state_temp_h, prev_coverage_vec,
               attn_state,  attn_values):
    """Hypothesis constructor.
    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
    """
    self.tokens = tokens
    self.probs = probs
    self.state_h = state_h
    self.state_c = state_c
    self.state_temp_h = state_temp_h
    self.state_temp_c = state_temp_c
    self.prev_coverage_vec = prev_coverage_vec
    self.attn_state = attn_state
    self.attn_dists = attn_values

  def extend(self, token, prob, state_h, state_c, state_temp_c, state_temp_h, prev_coverage_vec,
            attn_state, attn_values):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.
    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      probs = self.probs + [prob],
                      attn_state = attn_state,
                      attn_values = self.attn_dists + [attn_values],
                      state_h = state_h,
                      state_c = state_c,
                      state_temp_h = state_temp_h,
                      state_temp_c = state_temp_c,
                      prev_coverage_vec = prev_coverage_vec )

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(np.log(self.probs))

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return np.sum(np.log(self.probs))/len(self.tokens)


class Hypothesis_2nd(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, probs, state_h, state_c, state_temp_c, state_temp_h, prev_coverage_vec, 
               prev_coverage_between_decoders, prev_coverage_vec_different, 
               combined_attn_state, attns_state_different, attn_values):
    """Hypothesis constructor.
    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
    """
    self.tokens = tokens
    self.probs = probs
    self.state_h = state_h
    self.state_c = state_c
    self.state_temp_h = state_temp_h
    self.state_temp_c = state_temp_c
    self.prev_coverage_vec = prev_coverage_vec
    self.prev_coverage_vec_different = prev_coverage_vec_different
    self.prev_coverage_between_decoders = prev_coverage_between_decoders
    self.combined_attn_state = combined_attn_state
    self.attn_dists = attn_values
    self.attns_state_different = attns_state_different

  def extend(self, token, prob, state_h, state_c, state_temp_c, state_temp_h, 
            prev_coverage_vec, prev_coverage_between_decoders, prev_coverage_vec_different,
            attns_state_different,  combined_attn_state, attn_values):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.
    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis_2nd(tokens = self.tokens + [token],
                      probs = self.probs + [prob],
                      combined_attn_state = combined_attn_state,
                      attn_values = self.attn_dists + [attn_values],
                      state_h = state_h,
                      state_c = state_c,
                      state_temp_h = state_temp_h,
                      state_temp_c = state_temp_c,
                      prev_coverage_vec = prev_coverage_vec,
                      prev_coverage_between_decoders = prev_coverage_between_decoders,
                      prev_coverage_vec_different = prev_coverage_vec_different,
                      attns_state_different = attns_state_different
                       )

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(np.log(self.probs))

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return np.sum(np.log(self.probs))/len(self.tokens)
