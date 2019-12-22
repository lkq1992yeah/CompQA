class QBaseModule(object):

    def __init__(self, q_max_len, dim_wd_emb, dim_q_hidden):
        self.q_max_len = q_max_len
        self.dim_wd_emb = dim_wd_emb
        self.dim_q_hidden = dim_q_hidden

    # Input:
    #   q_embedding: (batch, q_max_len, dim_wd_emb)
    #   q_len: (batch, ) as int32
    # Output:
    #   q_hidden: (batch, q_max_len, dim_q_hidden)
    def forward(self, q_embedding, q_len, reuse=None):
        pass
