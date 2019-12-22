class SkBaseModule(object):

    def __init__(self, path_max_len, dim_item_hidden, dim_kb_emb, dim_sk_hidden):
        self.path_max_len = path_max_len
        self.dim_item_hidden = dim_item_hidden
        self.dim_kb_emb = dim_kb_emb
        self.dim_sk_hidden = dim_sk_hidden

    # Input:
    #   path_wd_hidden: (batch, path_max_len, dim_item_hidden)
    #   path_kb_hidden: (batch, path_max_len, dim_kb_emb)
    #   path_len: (batch, ) as int32
    #   focus_wd_hidden: (batch, dim_item_hidden)
    #   focus_kb_hidden: (batch, dim_kb_emb)
    # Output:
    #   sk_hidden: (batch, dim_sk_hidden)
    def forward(self, path_wd_hidden, path_kb_hidden, path_len, focus_wd_hidden, focus_kb_hidden, reuse=None):
        pass
